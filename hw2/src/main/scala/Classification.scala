import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, OneHotEncoderEstimator, StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{array, col, explode, max, min, sum, udf}
import org.apache.spark.sql.types.DoubleType

object Classification {

  val spark: SparkSession = SparkSession.builder().appName("Classifier")
    .config("spark.driver.maxResultSize", "5g")
    .config("spark.sql.shuffle.partitions", "5")
    .config("spark.driver.memory", "3g")
    .config("spark.executor.memory ", "3g")
    .config("spark.memory.offHeap.size", "4g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.master", "local").getOrCreate()

  def main(args: Array[String]): Unit = {

    val testPath = "./mlboot_test.tsv" // 6MB
    val trainPath = "./mlboot_train_answers.tsv" // 15 MB

    val loadDF = loadData()
    
    val testData = joinDF(testPath, loadDF)
    val trainData = joinDF(trainPath, loadDF)
      .withColumn("target", col("target").cast(DoubleType))

    classification(testData, trainData)

    spark.stop()
  }

  def classification(testDF: DataFrame,
                     trainDF: DataFrame): Unit = {

    val oneHotEncoderEstimator = new OneHotEncoderEstimator()
      .setInputCols(Array("cat_features"))
      .setOutputCols(Array("cat_vector"))

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("cat_vector", "date_diff", "vectors_features"))
      .setOutputCol("features")

    val selector = new ChiSqSelector()
      .setFdr(0.1)
      .setFeaturesCol("features")
      .setLabelCol("target")
      .setOutputCol("selectedFeatures")

    val scaler = new StandardScaler()
      .setInputCol("selectedFeatures")
      .setOutputCol("nn_features")
      .setWithStd(true)
      .setWithMean(true)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("target")
      .setMetricName("areaUnderROC")

    val layersVal1 = Array[Int](4, 5, 4, 3)
    val layersVal2 = Array[Int](3, 4, 3, 3)

    // create the trainer and set its parameters
    val neuralNetwork = new MultilayerPerceptronClassifier()
      .setSeed(1234L)
      .setLabelCol("target")
      .setFeaturesCol("nn_features")

    val paramGrid = new ParamGridBuilder()
      .addGrid(neuralNetwork.maxIter, Array(5, 10, 15))
      .addGrid(neuralNetwork.blockSize, Array(32, 48, 60))
      .addGrid(neuralNetwork.layers, Array(layersVal1, layersVal2))
      .build()

    val pipeline = new Pipeline()
      .setStages(Array(oneHotEncoderEstimator, vectorAssembler, selector, scaler, neuralNetwork))

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel = crossValidator.fit(trainDF)

    val cvPredictionDF = cvModel.transform(testDF)

    val accuracy = evaluator.evaluate(cvPredictionDF)

    println("Accuracy (ROC) with cross validation = " + accuracy)
  }

  def joinDF(path: String,
             forJoinDF: DataFrame): DataFrame = {
    val tempDataDF = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .load(path)

    val joinedDF =
      tempDataDF.join(forJoinDF, Seq("cuid"), "inner")
        .drop("cuid")

    joinedDF.printSchema()

    joinedDF
  }

  def loadData(): DataFrame = {

    val path = "./mlboot_data.tsv" // 10GB

    import spark.implicits._
    import org.apache.spark.sql.types._

    val schema = StructType(Array(
      StructField("cuid", StringType, nullable = true),
      StructField("cat_feat", IntegerType, nullable = true),
      StructField("feature_1", StringType, nullable = true),
      StructField("feature_2", StringType, nullable = true),
      StructField("feature_3", StringType, nullable = true),
      StructField("date_diff", LongType, nullable = true))
    )

    val dataDF = spark.read.format("csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .schema(schema)
      .csv(spark.sparkContext.textFile(path, 1000).toDS())

    val combineMaps = new CombineMaps[Int, Double](IntegerType, DoubleType, _ + _)

    val df = dataDF
      .withColumn("features_j", array(dataDF("feature_1"), dataDF("feature_2"), dataDF("feature_3")))
      .withColumn("features_json", explode(col("features_j")))
      .rdd.map(row => {
      import org.json4s._
      import org.json4s.jackson.JsonMethods._
      implicit val formats: DefaultFormats.type = org.json4s.DefaultFormats
      val cuid: String = row.getAs[String]("cuid")
      val cat_feat: Int = row.getAs[Int]("cat_feat")
      val jsonString: String = row.getAs[String]("features_json")
      val map: Map[Int, Double] = parse(jsonString).extract[Map[Int, Double]]
      val dateDiff: Long = row.getAs[Long]("date_diff")
      (cuid, cat_feat, map, dateDiff)
    }).toDF("cuid", "cat_features", "map_features", "dt_diff")
      .groupBy("cuid")
      .agg(max("cat_features") as "cat_features", min("dt_diff") as "date_diff", combineMaps(col("map_features")))

      /*
        * root
        * |-- cuid: string (nullable = true)
        * |-- cat_features: array (nullable = true)
        * |    |-- element: integer (containsNull = true)
        * |-- date_diff: long (nullable = true)
        * |-- combinemaps(map_features): map (nullable = true)
        * |    |-- key: integer
        * |    |-- value: double (valueContainsNull = true)
      */
      .withColumn("vectors_features", mapToSparse(col("combinemaps(map_features)")))
      .drop("combinemaps(map_features)")

    df.printSchema()

    df
  }

  def mapToSparse: UserDefinedFunction = udf((map: Map[Int, Double]) => {
    Vectors.sparse(map.max._1 + 1, map.toSeq)
  })
}

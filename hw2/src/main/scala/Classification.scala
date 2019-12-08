import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, LabeledPoint, OneHotEncoderEstimator, StandardScaler, VectorAssembler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{array, col, explode, udf}
import org.apache.spark.sql.types.DoubleType

object Classification {

  val spark: SparkSession = SparkSession.builder().appName("Classifier")
    .config("spark.driver.maxResultSize", "5g")
    .config("spark.driver.memory", "3g")
    .config("spark.executor.memory ", "3g")
    .config("spark.memory.offHeap.size", "4g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.master", "local").getOrCreate()

  def main(args: Array[String]): Unit = {

    val testPath = "./mlboot_test.tsv" // 6MB
    val trainPath = "./mlboot_train_answers.tsv" // 15 MB

    import spark.implicits._

    val dataDF = loadDF()

    val testDf = joinDF(testPath, dataDF)

    val trainDf = joinDF(trainPath, dataDF) // with labels
      .withColumnRenamed("target", "label")
      .withColumn("label", col("label").cast(DoubleType))
      .rdd.map(row => {
      val label: Double = row.getAs[Double]("label")
      val features: Vector = row.getAs[Vector]("features")
      LabeledPoint(label, features)
    }).toDF()

    classification(testDf, trainDf)

    spark.stop()
  }

  def classification(testDF: DataFrame,
                     trainDF: DataFrame): Unit = {

    testDF.printSchema()
    trainDF.printSchema()

    val selector = new ChiSqSelector()
      .setFdr(0.2)
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setOutputCol("selectedFeatures")

    val scaler = new StandardScaler()
      .setInputCol("selectedFeatures")
      .setOutputCol("rf_features")
      .setWithStd(true)
      .setWithMean(true)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("areaUnderROC")

    val randomForestClassifier = new RandomForestClassifier()
      .setFeaturesCol("rf_features")

    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForestClassifier.maxBins, Array(10, 15, 25))
      .addGrid(randomForestClassifier.maxDepth, Array(3, 5, 7))
      .addGrid(randomForestClassifier.numTrees, Array(9, 12, 16))
      .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
      .build()

    val pipeline = new Pipeline()
      .setStages(Array(selector, scaler, randomForestClassifier))

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel = crossValidator.fit(trainDF)

    val cvPredictionDF = cvModel.transform(testDF)

    val accuracy = evaluator.evaluate(cvPredictionDF)

    println("Accuracy (ROC) with cross validation = " + accuracy) // accuracy (ROC) is
  }

  def loadDF(): DataFrame = {
    val dataPath = "./mlboot_data.tsv" // 11 GB

    import spark.implicits._
    import org.apache.spark.sql.types._

    val schema = StructType(Array(
      StructField("cuid", StringType, nullable = true),
      StructField("cat_feature", IntegerType, nullable = true),
      StructField("feature_1", StringType, nullable = true),
      StructField("feature_2", StringType, nullable = true),
      StructField("feature_3", StringType, nullable = true),
      StructField("dt_diff", LongType, nullable = true))
    )

    val dataDF = spark.read.format("csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .schema(schema)
      .csv(spark.sparkContext.textFile(dataPath, 1000).toDS())

    dataDF
  }

  def vectorSparse: UserDefinedFunction = udf((json: String) => {
    val map: collection.mutable.Map[Int, Double] = collection.mutable.Map()
    json.substring(1, json.length - 1).split(",").map(_.trim.replace("\"", ""))
      .foreach(string => {
        if (!string.equals("")) {
          val splitStr = string.split(":")
          val index = splitStr(0).toInt
          val value = splitStr(1).toDouble
          map(index) = value
        }
      })
    var size = 0
    if (map.nonEmpty) {
      size = map.keysIterator.reduceLeft((x, y) => if (x > y) x else y) + 1
    }
    Vectors.sparse(size, map.toSeq)
  })

  def joinDF(path: String,
             dataFrame: DataFrame): DataFrame = {

    val tempDataDF = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .load(path)
      .join(dataFrame, Seq("cuid"), "inner")

    val dataDF = tempDataDF
      .withColumn("features_j", array(tempDataDF("feature_1"), tempDataDF("feature_2"), tempDataDF("feature_3")))
      .withColumn("features_j", explode(col("features_j")))
      .withColumn("features_j", vectorSparse(col("features_j")))
      .drop("feature_1")
      .drop("feature_2")
      .drop("feature_3")

    val df = new OneHotEncoderEstimator()
      .setInputCols(Array("cat_feature"))
      .setOutputCols(Array("cat_vector"))
      .fit(dataDF)
      .transform(dataDF)
      .drop("cat_feature")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("dt_diff", "features_j", "cat_vector"))
      .setOutputCol("features")

    val asmDF = vectorAssembler.transform(df)
      .drop("features_j")
      .drop("cat_vector")
      .drop("dt_diff")
      .drop("cuid")

    asmDF.printSchema()

    asmDF
  }
}

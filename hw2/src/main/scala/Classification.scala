import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{Binarizer, ChiSqSelector, CountVectorizer, Normalizer, StandardScaler, VectorAssembler, VectorSlicer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{array, avg, col, collect_list, explode, udf}
import org.apache.spark.sql.types.DoubleType

import scala.util.Sorting.quickSort
import scala.collection._
import org.apache.spark.sql.types._

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
      .withColumn("label", col("target").cast(DoubleType))
      .drop("target")


    testData.show(6, truncate = false)
    // trainData.show(6, truncate = false)

    classification(testData, trainData)

    spark.stop()
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

    import spark.implicits._

    val path = "./xaa.tsv" // 10GB

    val schema = StructType(Array(
      StructField("cuid", StringType, nullable = true),
      StructField("cat_feat", DoubleType, nullable = true),
      StructField("feature_1", StringType, nullable = true),
      StructField("feature_2", StringType, nullable = true),
      StructField("feature_3", StringType, nullable = true),
      StructField("date_diff", DoubleType, nullable = true))
    )

    val dataDF = spark.read.format("csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .schema(schema)
      .csv(spark.sparkContext.textFile(path, 500).toDS())

    val combineMaps = new CombineMaps[Int, Double](IntegerType, DoubleType, _ + _)

    val df = dataDF
      .withColumn("features_j", array(dataDF("feature_1"), dataDF("feature_2"), dataDF("feature_3")))
      .withColumn("features_json", explode(col("features_j")))
      .rdd.map(row => {
      import org.json4s._
      import org.json4s.jackson.JsonMethods._
      implicit val formats: DefaultFormats.type = org.json4s.DefaultFormats
      val cuid: String = row.getAs[String]("cuid")
      val cat_feat: Double = row.getAs[Double]("cat_feat")
      val jsonString: String = row.getAs[String]("features_json")
      val map: Map[Int, Double] = parse(jsonString).extract[Map[Int, Double]]
      val dateDiff: Double = row.getAs[Double]("date_diff")
      (cuid, cat_feat, map, dateDiff)
    }).toDF("cuid", "cat_features", "map_features", "dats_diff")
      .groupBy("cuid")
      .agg(
        collect_list("cat_features") as "cat_array",
        avg("dats_diff") as "dt_diff",
        combineMaps(col("map_features")))
      .withColumn("sparse_vector", mapToSparse(col("combinemaps(map_features)")))
      .withColumn("cat_vector", convertArrayToVector(col("cat_array")))
      .drop("combinemaps(map_features)")
      .drop("cat_array")

    df
  }

  def classification(testDF: DataFrame,
                     trainDF: DataFrame): Unit = {

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("areaUnderROC")

    val binarizer = new Binarizer()
      .setInputCol("sparse_vector")
      .setOutputCol("binarized_vector_features")
      .setThreshold(4)

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("dt_diff", "cat_vector", "binarized_vector_features"))
      .setOutputCol("rf_features")

    val chiSqSelector = new ChiSqSelector()
      .setLabelCol("label")
      .setFeaturesCol("rf_features")
      .setOutputCol("features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaled_features")
      .setWithMean(true)
      .setWithStd(true)

    // create the trainer and set its parameters
    val randomForestClassifier = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("scaled_features")

    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForestClassifier.maxBins, Array(15, 25, 35, 45))
      .addGrid(randomForestClassifier.maxDepth, Array(4, 6, 8, 10, 12, 14, 16))
      .addGrid(randomForestClassifier.numTrees, Array(12, 15, 18, 20))
      .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
      .build()

    val pipeline = new Pipeline()
      .setStages(Array(binarizer, vectorAssembler, chiSqSelector, scaler, randomForestClassifier))

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

  def convertArrayToVector: UserDefinedFunction =
    udf((features: mutable.WrappedArray[Double]) => Vectors.dense(features.toArray))

  def mapToSparse: UserDefinedFunction =
    udf((map: Map[Int, Double]) => Vectors.sparse(map.keys.max + 1, map.toSeq))
}

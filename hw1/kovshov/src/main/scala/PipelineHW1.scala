import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StandardScaler, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}


object PipelineHW1 {
  private val selector = new ChiSqSelector()
    .setFpr(0.1)
    .setFeaturesCol("vectorizedFeatures")
    .setLabelCol("label")
    .setOutputCol("selectedFeatures")

  private val scaler = new StandardScaler()
    .setInputCol("selectedFeatures")
    .setOutputCol("features")
    .setWithStd(true)
    .setWithMean(true)

  private val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = createSparkSession()

    val data: DataFrame = spark
      .read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("./ml_dataset.csv")

    val Array(trainData, testData) = data.randomSplit(Array(0.7, 0.3))

    val vectorAssembler = new VectorAssembler()
      .setInputCols(trainData.columns.dropRight(1))
      .setOutputCol("vectorizedFeatures")

    val regressionPipeline = logisticRegression(vectorAssembler)
    val forestPipeline = randomForest(vectorAssembler)

    println("Logistic regression pipeline")
    val regressionAUC = workPipeline(regressionPipeline, trainData, testData)

    println("Random forest pipeline")
    val forestAUC = workPipeline(forestPipeline, trainData, testData)

    println("Logistic regression AUC = " + regressionAUC)
    println("Random forest AUC = " + forestAUC)
  }

  def logisticRegression(vectorAssembler: VectorAssembler): Pipeline = {
    val logisticRegression = new LogisticRegression()
      .setMaxIter(10)
      .setElasticNetParam(0.8)
      .setRegParam(0.3)
      .setFeaturesCol("features")
      .setLabelCol("label")

    createPipeline(Array(vectorAssembler, selector, scaler, logisticRegression))
  }

  def randomForest(vectorAssembler: VectorAssembler): Pipeline = {
    val randomForest = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)

    createPipeline(Array(vectorAssembler, selector, scaler, randomForest))
  }

  def workPipeline(pipeline: Pipeline, trainData: DataFrame, testData: DataFrame): Double = {
    val predict = pipeline.fit(trainData).transform(testData)

    predict.select("label", "prediction", "features").show(15)
    evaluator.evaluate(predict)
  }

  def createPipeline(array: Array[PipelineStage]): Pipeline = new Pipeline().setStages(array)

  def createSparkSession(): SparkSession = {
    val spark = SparkSession.builder
      .appName("Pipelines")
      .master("local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("OFF")
    spark
  }
}

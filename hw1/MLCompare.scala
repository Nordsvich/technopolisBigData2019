
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StandardScaler, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object MLCompare {

  private val CSV_PATH: String = "../scala/src/main/hw1/ml_dataset.csv"
  private val VECTORIZED_COL: String = "vectorizedFeatures"
  private val FEATURES_COL: String = "features"
  private val LABEL_COL: String = "label"
  private val SCALED_COL: String = "scaledFeatures"
  private val RANDOM_SEED: Int = 5043
  private val PREDICTION_COL: String = "prediction"

  def main(args: Array[String]): Unit = {
    val sparkSession: SparkSession = SparkSession.builder()
      .appName("ML_AUC_COMPARE")
      .config("spark.master", "local")
      .getOrCreate()

    sparkSession.sparkContext.setLogLevel("OFF")

    val dataset: DataFrame = sparkSession.read.format("csv")
      .option("header", value = "true")
      .option("inferSchema", value = "true")
      .load(CSV_PATH)

    val Array(training, testDataset) = dataset.randomSplit(Array(0.7, 0.3), RANDOM_SEED)

    val vectorAssembler = new VectorAssembler()
      .setInputCols(training.columns.dropRight(1)) // убираем `label` и смотрим только на `future`
      .setOutputCol(VECTORIZED_COL)

    val chiSqSelector = new ChiSqSelector()
      .setFpr(0.1)
      .setFeaturesCol(VECTORIZED_COL)
      .setLabelCol(LABEL_COL)
      .setOutputCol(FEATURES_COL)

    val standardScaler = new StandardScaler()
      .setInputCol(FEATURES_COL)
      .setOutputCol(SCALED_COL)
      .setWithStd(true)
      .setWithMean(true)

    //Random forest
    val rf = new RandomForestClassifier()
      .setLabelCol(LABEL_COL)
      .setFeaturesCol(SCALED_COL)
      .setNumTrees(100)

    //LogisticRegression
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setElasticNetParam(0.8)
      .setRegParam(0.3)
      .setFeaturesCol(SCALED_COL)
      .setLabelCol(LABEL_COL)

    val lrPipeline = pipelineOf(Array(vectorAssembler, chiSqSelector, standardScaler, lr))
    val lrPredict = getPrediction(lrPipeline, training, testDataset)

    println("Logistic regression: ")
    lrPredict.select(LABEL_COL, PREDICTION_COL, FEATURES_COL).show(30)

    val rfPipeline = pipelineOf(Array(vectorAssembler, chiSqSelector, standardScaler, rf))
    val rfPredict = getPrediction(rfPipeline, training, testDataset)

    println("Random forest: ")
    rfPredict.select(LABEL_COL, PREDICTION_COL, FEATURES_COL).show(30)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol(LABEL_COL)
      .setRawPredictionCol(PREDICTION_COL)
      .setMetricName("areaUnderROC")

    println("Logistic regression AUC: " + evaluator.evaluate(lrPredict))
    println("Random forest AUC: " + evaluator.evaluate(rfPredict))

    sparkSession.stop()
  }

  def getPrediction(pipeline: Pipeline, training: Dataset[Row], testDataSet: Dataset[Row]): DataFrame = {
    //Fit the model
    val model = pipeline.fit(training)
    model.transform(testDataSet)
  }

  def pipelineOf(array: Array[PipelineStage]): Pipeline = new Pipeline().setStages(array)
}

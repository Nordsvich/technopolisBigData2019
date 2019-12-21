import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, ChiSqSelectorModel, StandardScaler, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.sql.{Dataset, Row, SparkSession}

object Main {
  val path = "../ml_dataset.csv"

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = createSparkSession()

    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)
      .cache()
    val features = df.columns.dropRight(1)
    val Array(training, test) = df.randomSplit(Array(0.8, 0.2), seed = 42)

    println("Random forest")
    val rf = new RandomForestClassifier()
    val rfParams = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(10, 15, 20))
      .addGrid(rf.impurity, Array("gini", "entropy"))
    val (rfPipelineMode, rfModel) = createModel(rf, features, training, rfParams)
    println(s"maxDepth = ${rfModel.getMaxDepth}, impurity = ${rfModel.getImpurity}")
    evaluateModel(rfPipelineMode, test)

    println("Linear regression")
    val lr = new LogisticRegression()
      .setMaxIter(20)
    val lrParams = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01, 0.001))
      .addGrid(lr.elasticNetParam, Array(0.5, 0.7, 0.9))
    val (pipelineLrModel, lrModel) = createModel(lr, features, training, lrParams)
    println(s"regParam = ${lrModel.getRegParam}, elasticNetParam = ${lrModel.getElasticNetParam}")
    evaluateModel(pipelineLrModel, test)
  }

  def createModel[F, M <: ProbabilisticClassificationModel[F, M]](classifier: ProbabilisticClassifier[F, _, M],
                                                                  features: Array[String],
                                                                  train: Dataset[Row],
                                                                  params: ParamGridBuilder): (PipelineModel, M) = {
    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val selector = new ChiSqSelector()
      .setFeaturesCol("features")
      .setLabelCol("label")
      .setOutputCol("topFeatures")

    val scaler = new StandardScaler()
      .setInputCol("topFeatures")
      .setOutputCol("scaledFeatures")

    classifier.setFeaturesCol("scaledFeatures")

    val pipeline = new Pipeline()
      .setStages(Array(assembler, selector, scaler, classifier))

    val paramGrid = params
      .addGrid(selector.numTopFeatures, 20 to 100 by 20)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(4)
      .setParallelism(2)

    val model = cv.fit(train)
    val pipelineModel = model.bestModel.asInstanceOf[PipelineModel]
    println(s"numTopFeatures = ${pipelineModel.stages(1).asInstanceOf[ChiSqSelectorModel].getNumTopFeatures}")

    (pipelineModel, pipelineModel.stages(3).asInstanceOf[M])
  }

  def evaluateModel[T <: Transformer](model: T, test: Dataset[Row]): Unit = {
    val predictions = model.transform(test)
    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol("prediction")
      .setLabelCol("label")

    for (metric <- Seq("areaUnderROC", "areaUnderPR")) {
      println(s"$metric = ${evaluator.setMetricName(metric).evaluate(predictions)}")
    }
  }

  def createSparkSession(): SparkSession = {
    val spark = SparkSession.builder
      .appName("Spark ML")
      .master("local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark
  }
}

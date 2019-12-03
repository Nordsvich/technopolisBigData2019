import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StandardScaler, VectorAssembler}
import org.apache.spark.sql.SparkSession

object Main extends App {
  val spark = SparkSession.builder
    .master("local[*]")
    .appName("BigData2019")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  val data = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("ml_dataset.csv")

  data.printSchema()

  // Prepare training and test data
  val seed = 1234
  val Array(training, test) = data.randomSplit(Array(0.7, 0.3), seed)

  // Creating features column
  val assembler = new VectorAssembler()
    .setInputCols(training.columns.dropRight(1))
    .setOutputCol("featuresCol")

  val selector = new ChiSqSelector()
    .setFeaturesCol("featuresCol")
    .setLabelCol("label")
    .setOutputCol("selectedFeatures")

  val scaler = new StandardScaler()
    .setInputCol("selectedFeatures")
    .setOutputCol("features")
    .setWithStd(true)
    .setWithMean(true)

  val logisticRegression = new LogisticRegression()
    .setMaxIter(100)
    .setRegParam(0.02)
    .setElasticNetParam(0.8)

  // Creating pipeline
  val lrPipeline = new Pipeline().setStages(Array(assembler, selector, scaler, logisticRegression))
  val lrModel = lrPipeline.fit(training)

  // Test model with test data
  val lrPrediction = lrModel.transform(test)
  lrPrediction.select("label", "prediction", "features").show(10)

  // Train Random Forest model with training data set
  val randomForestClassifier = new RandomForestClassifier()
    .setImpurity("gini")
    .setMaxDepth(3)
    .setNumTrees(20)
    .setFeatureSubsetStrategy("auto")
    .setSeed(seed)

  // Creating pipeline
  val rfPipeline = new Pipeline().setStages(Array(assembler, selector, scaler, randomForestClassifier))
  val rfModel = rfPipeline.fit(training)

  // Test model with test data
  val rfPrediction = rfModel.transform(test)
  rfPrediction.select("label", "prediction", "features").show(10)

  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")

  // Measure the accuracy
  val lrAccuracy = evaluator.evaluate(lrPrediction)
  println("Logistic Regression accuracy: " + lrAccuracy)

  val rfAccuracy = evaluator.evaluate(rfPrediction)
  println("Random Forest Classifier accuracy: " + rfAccuracy)
}

import org.apache.spark.ml.{Pipeline}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StandardScaler, VectorAssembler}
import org.apache.spark.sql.SparkSession

object HomeWork {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("HomeWork")
      .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")
    val df = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("ml_dataset.csv")

    val Array(training, test) = df.randomSplit(Array(0.75, 0.25))

    val assembler = new VectorAssembler()
      .setInputCols(training.columns.dropRight(1))
      .setOutputCol("vectorizedFeatures")

    val selector = new ChiSqSelector()
      .setFpr(0.1)
      .setFeaturesCol("vectorizedFeatures")
      .setLabelCol("label")
      .setOutputCol("selectedFeatures")

    val scaler = new StandardScaler()
      .setInputCol("selectedFeatures")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setElasticNetParam(0.8)
      .setRegParam(0.3)
      .setFeaturesCol("features")
      .setLabelCol("label")

    val lrPipeline = new Pipeline()
      .setStages(Array(assembler, selector, scaler, lr))

    val lrModel = lrPipeline.fit(training)

    val lrPredict = lrModel.transform(test)

    lrPredict.select("label", "prediction", "features").show(10)


    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")


    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val rfPipeline = new Pipeline()
      .setStages(Array(assembler, selector, scaler, rf))

    val rfModel = rfPipeline.fit(training)

    val rfPredict = rfModel.transform(test)

    rfPredict.select("label", "prediction", "features").show(10)

    println("LogisticRegression AUC - " + evaluator.evaluate(lrPredict))
    println("RandomForest AUC - " + evaluator.evaluate(rfPredict))
  }
}

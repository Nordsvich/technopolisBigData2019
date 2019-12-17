import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, VectorAssembler}
import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local")
      .appName("HW1")
      .getOrCreate()
    spark.sparkContext.setLogLevel("OFF")

    val df = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("ml_dataset.csv")

    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    val assembler = new VectorAssembler()
      .setInputCols(trainingData.columns.dropRight(1))
      .setOutputCol("vectorizedFeatures")

    val selector = new ChiSqSelector().setFpr(0.1)
      .setFeaturesCol("vectorizedFeatures")
      .setLabelCol("label")
      .setOutputCol("selectedFeatures")

    val forestClassifier = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)

    val forestPipeline = new Pipeline().setStages(Array(assembler, selector, forestClassifier))

    val gBTClassifier = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(10)

    val gBTPipeline = new Pipeline().setStages(Array(assembler, selector, gBTClassifier))

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")

    val forestModel = forestPipeline.fit(trainingData)
    val forestPredict = forestModel.transform(testData)
    println("Forest")
    forestPredict.select("label", "prediction", "features").show(10)
    val forestAUC = evaluator.evaluate(forestPredict)

    val gBTModel = gBTPipeline.fit(trainingData)
    val gBTPredict = gBTModel.transform(testData)
    println("GBT")
    gBTPredict.select("label", "prediction", "features").show(10)
    val gBTAUC = evaluator.evaluate(gBTPredict)

    println("RandomForestAUC: " + forestAUC)
    println("GBTRegressionAUC: " + gBTAUC)
  }
}

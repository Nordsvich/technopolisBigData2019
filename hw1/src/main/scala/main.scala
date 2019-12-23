import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.classification.{GBTClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object main {

  val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local")
      .appName("HW1")
      .config("spark.executor.cores","4")
      .config("spark.executor.memory","4G")
      .getOrCreate()
    spark.sparkContext.setLogLevel("OFF")

    val dataFrame = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("ml_dataset.csv")

    val Array(trainingData, testData) = dataFrame.randomSplit(Array(0.7, 0.3))

    val assembler = new VectorAssembler()
      .setInputCols(trainingData.columns.dropRight(1))
      .setOutputCol("vectorAssemblerFeauters")

    val selector = new ChiSqSelector()
      .setFeaturesCol(assembler.getOutputCol)
      .setLabelCol("label")
      .setOutputCol("features")

    //RandomForest
    val forestClassifier = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)
    val randomForestModel = createPipeline(Array(assembler, selector, forestClassifier))


    //GBTClassifier
    val gbtClassifier = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(50)
    val gbtModel = createPipeline(Array(assembler, selector, gbtClassifier))

    println("\nRandom Forest Regression")
    trainingANDtest(randomForestModel,trainingData,testData)

    println("\nGradient-boosted Tree Regression")
    trainingANDtest(gbtModel,trainingData,testData)

  }

  def createPipeline(array: Array[PipelineStage]): Pipeline = new Pipeline().setStages(array)

  def trainingANDtest (pipeline: Pipeline, trainingData : DataFrame,testData: DataFrame): Unit ={
    val model = pipeline.fit(trainingData)
    val predict = model.transform(testData);
    predict.select("label","prediction","features").show(5)
    val aus = evaluator.evaluate(predict)
    println("AUC for metod: " + aus)
  }
}


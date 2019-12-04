package igorlo

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StandardScaler, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object Homework {

  def main(args: Array[String]): Unit = {

    val spark: SparkSession = SparkSession.builder()
      .master("local")
      .appName("hw1")
      .getOrCreate()
    spark.sparkContext.setLogLevel("OFF")

    val data: DataFrame = spark
      .read
      .format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("hw1/ml_dataset.csv")

    val Array(trainData, testData) = data.randomSplit(Array(0.7, 0.3))

    val vectorAssembler = new VectorAssembler()
      .setInputCols(trainData.columns.dropRight(1))
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

    //--------------------------- LR ------------
    val logisticRegression = new LogisticRegression()
      .setMaxIter(10)
      .setElasticNetParam(0.8)
      .setRegParam(0.3)
      .setFeaturesCol("features")
      .setLabelCol("label")
    val regressionPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, selector, scaler, logisticRegression))
    val firstModel = regressionPipeline.fit(trainData)
    val firstPredict = firstModel.transform(testData)

    //--------------------------- RF ------------
    val randomForest = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)
    val forestPipeline = new Pipeline()
      .setStages(Array(vectorAssembler, selector, scaler, randomForest))
    val secondModel = forestPipeline.fit(trainData)
    val secondPredict = secondModel.transform(testData)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")

    //Выводим результат
    firstPredict.select("label", "prediction", "features").show(100)
    secondPredict.select("label", "prediction", "features").show(100)
    println("---------------")
    println("LR AUC")
    println(evaluator.evaluate(firstPredict))
    println("---------------")
    println("RF AUC")
    println(evaluator.evaluate(secondPredict))

  }

}


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StandardScaler, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

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

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(10)


    val rfPipeline = new Pipeline()
      .setStages(Array(assembler, selector, scaler, rf))


    val paramGridLr = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.8, 0.4, 0.1))
      .addGrid(lr.maxIter, Array(10, 15, 20))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val paramGridRf = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(5, 10, 15))
      .addGrid(rf.numTrees, Array(3, 6, 9))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")


    val trvLr = new TrainValidationSplit()
      .setEstimator(lrPipeline)
      .setEstimatorParamMaps(paramGridLr)
      .setEvaluator(evaluator)
      .setTrainRatio(0.75)

    val trvRf = new TrainValidationSplit()
      .setEstimator(rfPipeline)
      .setEstimatorParamMaps(paramGridRf)
      .setEvaluator(evaluator)
      .setTrainRatio(0.75)

    val lrModel = trvLr.fit(training)

    val rfModel = trvRf.fit(training)

    val lrPredict = lrModel.transform(test)
    val rfPredict = rfModel.transform(test)

    lrPredict.select("label", "prediction", "features").show(10)

    rfPredict.select("label", "prediction", "features").show(10)

    println("LogisticRegression AUC - " + evaluator.evaluate(lrPredict))
    println("RandomForest AUC - " + evaluator.evaluate(rfPredict))
  }
}

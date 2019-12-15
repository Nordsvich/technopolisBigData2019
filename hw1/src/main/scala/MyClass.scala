import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StandardScaler, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.SparkSession

object MyClass {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("elenapranova")
      .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")

    val dataMLDataset = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("ml_dataset.csv")

    val Array(training, test) = dataMLDataset.randomSplit(Array(0.7, 0.3), seed = 5)

    val assembler = new VectorAssembler()
      .setInputCols(dataMLDataset.columns.dropRight(1))
      .setOutputCol("assebledFeatures")

    val selector = new ChiSqSelector()
      .setFpr(0.05)
      .setFeaturesCol(assembler.getOutputCol)
      .setLabelCol("label")
      .setOutputCol("selectedFeatures")

    val scaler = new StandardScaler()
      .setInputCol(selector.getOutputCol)
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")

    val paramGridRF = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(5, 10, 15))
      .addGrid(rf.numTrees, Array(5, 10, 15))
      .build()

    val pipelineRF = new Pipeline()
      .setStages(Array(assembler, selector, scaler, rf))

    val cvRF = new TrainValidationSplit()
      .setEstimator(pipelineRF)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGridRF)
//      .setNumFolds(10)
      .setTrainRatio(0.8)
      .setParallelism(4)

    val rfModel = cvRF.fit(training)

    val lr = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol(selector.getOutputCol)

    val paramGridLR = new ParamGridBuilder()
      .addGrid(lr.maxIter, Array(10, 20))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val pipelineLR = new Pipeline().setStages(Array(assembler, selector, scaler, lr))

    val cvLR = new TrainValidationSplit()
      .setEstimator(pipelineLR)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGridLR)
//      .setNumFolds(10)
      .setTrainRatio(0.8)
      .setParallelism(4)

    val lrModel = cvLR.fit(training)

    val rfPredict = rfModel.transform(test)

    val lrPredict = lrModel.transform(test)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")

    println("Random Forest " + evaluator.evaluate(rfPredict))
    println("Logistic Regression " + evaluator.evaluate(lrPredict))
  }
}

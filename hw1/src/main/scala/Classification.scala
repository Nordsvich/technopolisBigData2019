import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, PCA}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row, SparkSession}


object Classification {

  def main(args: Array[String]): Unit = {

    val pathDataSetCSV = "./ml_dataset.csv"
    val spark = SparkSession.builder().appName("Classifier").config("spark.master", "local").getOrCreate()
    val rawString: RDD[String] = spark.read.format("csv").option("header", "true").load(pathDataSetCSV).rdd.map(_.mkString(","))

    import spark.implicits._

    val rowData = rawString.map(_.split(",")).map({
      csv =>
        val label = csv.last.toDouble
        val point = csv.init.map(_.toDouble)
        (label, point)
    })

    val rowDataset = rowData
      .map { case (label, point) =>
        LabeledPoint(label, Vectors.dense(point))
    }.toDS()

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pca_features")
      .setK(33)
      .fit(rowDataset)

    //features selection

    val dataset = pca.transform(rowDataset)
      .select("label", "pca_features")
      .withColumnRenamed("pca_features", "features")

    val seed = 1234L
    val Array(trainingData, testData) = dataset.randomSplit(Array(0.85, 0.15), seed)
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("areaUnderROC")

    randomForestClassification(trainingData, testData, evaluator)
    linearSupportVectorMachineClassification(trainingData, testData, evaluator)

    spark.stop()
  }


  def getCrossValidator(pipeline: Pipeline,
            paramGridBuilder: Array[ParamMap],
            evaluator: BinaryClassificationEvaluator): CrossValidator = {
    return new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGridBuilder)
      .setNumFolds(5)
  }

  def linearSupportVectorMachineClassification(trainingData : Dataset[Row],
                                               testData : Dataset[Row],
                                               evaluator: BinaryClassificationEvaluator): Unit = {
    //lsvm
    val lsvc = new LinearSVC()

    val randomForestModel = lsvc.fit(trainingData)

    val predictionDf = randomForestModel.transform(testData)

    val paramGrid = new ParamGridBuilder()
      .addGrid(lsvc.regParam, Array(0.1, 0.2, 0.3))
      .addGrid(lsvc.maxIter, Array(10, 15, 20))
      .build()


    val pipeline = new Pipeline()
      .setStages(Array(lsvc))

    val cv = getCrossValidator(pipeline, paramGrid, evaluator)

    val cvModel = cv.fit(trainingData)

    val cvPredictionDf = cvModel.transform(testData)

    val accuracy = evaluator.evaluate(predictionDf)
    println("Accuracy with lsvm (ROC) without cross validation = " + accuracy)

    val cvAccuracy = evaluator.evaluate(cvPredictionDf)
    println("Accuracy (ROC) with LSVM with cross-validation = " + cvAccuracy)

  }

  def randomForestClassification(trainingData : Dataset[Row],
                                 testData : Dataset[Row],
                                 evaluator: BinaryClassificationEvaluator): Unit = {

    // random forest

    val randomForestClassifier = new RandomForestClassifier()

    val pipeline = new Pipeline()
      .setStages(Array(randomForestClassifier))

    val randomForestModel = randomForestClassifier.fit(trainingData)

    val predictionDf = randomForestModel.transform(testData)

    //Hyperparameters selection

    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForestClassifier.maxBins, Array(25, 28, 31))
      .addGrid(randomForestClassifier.maxDepth, Array(4, 6, 8))
      .addGrid(randomForestClassifier.numTrees, Array(12, 15, 18))
      .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
      .build()


    val cv = getCrossValidator(pipeline, paramGrid, evaluator)

    val cvModel = cv.fit(trainingData)

    val cvPredictionDf = cvModel.transform(testData)

    val accuracy = evaluator.evaluate(predictionDf)
    println("Accuracy with random forest (ROC) without cross validation = " + accuracy)

    val cvAccuracy = evaluator.evaluate(cvPredictionDf)
    println("Accuracy (ROC) with cross validation = " + cvAccuracy)
  }
}
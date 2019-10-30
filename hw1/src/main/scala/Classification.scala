import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, PCA}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


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

    val dataset = pca.transform(rowDataset)
      .select("label", "pca_features")
      .withColumnRenamed("pca_features", "features")

    val seed = 1234L
    val Array(trainingData, testData) = dataset.randomSplit(Array(0.85, 0.15), seed)

    val randomForestClassifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(3)
      .setNumTrees(20)
      .setFeatureSubsetStrategy("auto")
      .setSeed(seed)

    val stages = Array(randomForestClassifier)

    val pipeline = new Pipeline()
      .setStages(stages)

    val pipelineModel = pipeline.fit(trainingData)

    val pipelinePredictionDf = pipelineModel.transform(testData)
    pipelinePredictionDf.show(10)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("areaUnderROC")

    val randomForestModel = randomForestClassifier.fit(trainingData)

    val predictionDf = randomForestModel.transform(testData)

    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForestClassifier.maxBins, Array(25, 28, 31))
      .addGrid(randomForestClassifier.maxDepth, Array(4, 6, 8))
      .addGrid(randomForestClassifier.impurity, Array("entropy", "gini"))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val cvModel = cv.fit(trainingData)

    // test cross validated model with test data
    val cvPredictionDf = cvModel.transform(testData)


    val accuracy = evaluator.evaluate(predictionDf)
    println("accuracy with random forest (ROC) = " + accuracy)
    val pipelineAccuracy = evaluator.evaluate(pipelinePredictionDf)
    println("pipelineAccuracy (ROC) = " + pipelineAccuracy)
    val cvAccuracy = evaluator.evaluate(cvPredictionDf)
    println("cvAccuracy (ROC) = " + cvAccuracy)


    spark.stop()
  }
}
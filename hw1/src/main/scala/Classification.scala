import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.feature.PCA
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, SparkSession}


object Classification {

  def main(args: Array[String]): Unit = {

    val pathDataSetCSV = "./ml_dataset.csv"
    val spark = SparkSession.builder().appName("Classifier").config("spark.master", "local").getOrCreate()
    val rawString: RDD[String] = spark.read.format("csv").option("header", "true").load(pathDataSetCSV).rdd.map(_.mkString(","))

    import spark.implicits._

    val dataRaw = rawString.map(_.split(",")).map({
      csv =>
        val label = csv.last.toDouble
        val point = csv.init.map(_.toDouble)
        (label, point)
    })

    val data: RDD[LabeledPoint] = dataRaw
      .map { case (label, point) =>
        LabeledPoint(label, Vectors.dense(point))
      }

    val pca = new PCA(33).fit(data.map(_.features))

    val dataset = data.map(point => point.copy(features = pca
      .transform(point.features))).toDS()

    val Array(training: Dataset[LabeledPoint], test: Dataset[LabeledPoint]) =
      dataset.randomSplit(Array(0.85, 0.15), seed = 1234L)

    
    //  DecisionTree

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 12
    val maxBins = 35

    val decisionTreeModel = DecisionTree.trainClassifier(training.rdd,
      numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)


    val predictLabels = test.map { point =>
      val prediction = decisionTreeModel.predict(point.features)
      (point.label, prediction)
    }

    val decisionTreeMetrics = new BinaryClassificationMetrics(predictLabels.rdd)
    val decisionTreeMetricsAUROC = decisionTreeMetrics.areaUnderROC()

    println(s"Area under ROC with decision tree = $decisionTreeMetricsAUROC")


    //Linear Support Vector Machines

    val numIterations = 40
    val svmModel = SVMWithSGD.train(training.rdd, numIterations)

    // Clear the default threshold.
    svmModel.clearThreshold()

    val scoreAndLabels = test.map { point =>
      val prediction = svmModel.predict(point.features)
      (point.label, prediction)
    }

    // Get evaluation metrics.
    val svmMetrics = new BinaryClassificationMetrics(scoreAndLabels.rdd)
    val auROCLSVM = svmMetrics.areaUnderROC()

    println(s"Area under ROC with svm = $auROCLSVM")

    spark.stop()
  }
}
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

    val data: Dataset[LabeledPoint] = dataRaw
      .map { case (label, point) =>
        LabeledPoint(label, Vectors.dense(point))
      }.toDS()

    val Array(training: Dataset[LabeledPoint], test: Dataset[LabeledPoint]) =
      data.randomSplit(Array(0.85, 0.15), seed = 1234L)

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxDepth = 15
    val maxBins = 35

    val model = DecisionTree.trainClassifier(training.rdd, numClasses, categoricalFeaturesInfo,
      impurity, maxDepth, maxBins)

    //Task classification
    // Evaluate model on test instances and compute test error
    val predictLabels = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    //predictLabels.show()

    val testErr = predictLabels.filter(
      row => row._1 != row._2
    ).count().toDouble / test.count()
    println("Test Error = " + testErr)
   // println("Learned classification tree model:\n" + model.toDebugString)

    spark.stop()
  }
}

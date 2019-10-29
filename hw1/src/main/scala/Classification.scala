import org.apache.spark.ml.Pipeline
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, LabeledPoint}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}


object Classification {

    def main(args: Array[String]): Unit = {


      val pathDataSetCSV = "./ml_dataset.csv"
      val spark = SparkSession.builder().appName("Classifier").config("spark.master", "local").getOrCreate()
      var watherRaw: RDD[String] = spark.read.format("csv").option("header", "true").load(pathDataSetCSV).rdd.map(_.mkString(","))

      import spark.implicits._

      val dataRaw = watherRaw.map(_.split(",")).map({
          csv => val label = csv.last.toDouble
          val point = csv.init.map(_.toDouble)
            (label, point)
        }
      )
      val data: Dataset[LabeledPoint] = dataRaw
        .map { case (label, point) =>
          LabeledPoint(label, Vectors.dense(point))
        }.toDS()

      val Array(training: Dataset[LabeledPoint], test: Dataset[LabeledPoint]) =
        data.randomSplit(Array(0.85, 0.15), seed = 1234L)
      

      spark.stop()
    }
}

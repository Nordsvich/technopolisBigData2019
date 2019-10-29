import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

object Classification extends App {

      val spark = SparkSession.builder().appName("Classifier").config("spark.master", "local").getOrCreate()

      val pathDataSetCSV = "./ml_dataset.csv"

      var csvDataFrame = spark.read.format("csv").option("header", "true").load(pathDataSetCSV)
      val featuresDataFrame = csvDataFrame.drop("label")
      val labelDataFrame = csvDataFrame.select("label")
      csvDataFrame = null

      spark.stop();

}
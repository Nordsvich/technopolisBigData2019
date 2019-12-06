import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext, SparkSession}
import org.apache.spark.sql.functions.broadcast


object Classification {

  def main(args: Array[String]): Unit = {

    val dataPath = "./mlboot_data.tsv"
    val testPath = "./mlboot_test.tsv"
    val trainPath = "./mlboot_train_answers.tsv"


    val spark = SparkSession.builder().appName("Classifier")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.sql.broadcastTimeout", "36000")
      .config("spark.master", "local").getOrCreate()

    val dataDF = spark.read.format("csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .load(dataPath)
      .withColumnRenamed("_c0", "cuid")
      .withColumnRenamed("_c1", "cat_feature")
      .withColumnRenamed("_c2", "feature_1")
      .withColumnRenamed("_c3", "feature_2")
      .withColumnRenamed("_c4", "feature_3")
      .withColumnRenamed("_c5", "dt_diff")
      .repartition(2000)


    val testDF = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .load(testPath)

    val trainDF = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .load(trainPath)
      .join(broadcast(dataDF),Seq("cuid"),"inner")
      .show(6, truncate = false)

    spark.stop()
  }
}

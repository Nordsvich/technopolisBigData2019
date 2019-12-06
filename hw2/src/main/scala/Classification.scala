import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext, SparkSession}
import org.apache.spark.sql.functions.broadcast


object Classification {

  def main(args: Array[String]): Unit = {

    val dataPath = "./mlboot_data.tsv" // 11 GB
    val testPath = "./mlboot_test.tsv" // 6MB
    val trainPath = "./mlboot_train_answers.tsv" // 15 MB

    val spark = SparkSession.builder().appName("Classifier")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.sql.broadcastTimeout", "36000")
      .config("spark.master", "local").getOrCreate()

    import spark.implicits._
    import org.apache.spark.sql.types._

    val schema = StructType(Array(
      StructField("cuid", StringType, nullable = true),
      StructField("cat_feature", LongType, nullable = true),
      StructField("feature_1", StringType, nullable = true),
      StructField("feature_2", StringType, nullable = true),
      StructField("feature_3", StringType, nullable = true),
      StructField("dt_diff", LongType, nullable = true))
    )

    val dataDF = spark.read.format("csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .schema(schema)
      .csv(spark.sparkContext.textFile(dataPath, 500).toDS())

    val testDF = joinDf(testPath, spark, dataDF)
    val trainDF = joinDf(trainPath, spark, dataDF)

    spark.stop()
  }

  def joinDf(path: String,
             spark: SparkSession,
             dataDF: DataFrame): DataFrame = {
    val testDF = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .load(path)
      .join(dataDF, Seq("cuid"), "inner")
    testDF;
  }
}

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.sql.{DataFrame, Dataset, Row, SQLContext, SparkSession}

object Classification {

  def main(args: Array[String]): Unit = {

    val testPath = "./mlboot_test.tsv" // 6MB
    val trainPath = "./mlboot_train_answers.tsv" // 15 MB

    val spark = SparkSession.builder().appName("Classifier")
      .config("spark.driver.maxResultSize", "2g")
      .config("spark.master", "local").getOrCreate()


    val dataDF = loadDF(spark)

    val testDf = joinDF(testPath, dataDF, spark)
    val trainDf = joinDF(trainPath, dataDF, spark)

    spark.stop()
  }

  def loadDF(spark: SparkSession): DataFrame = {
    val dataPath = "./mlboot_data.tsv" // 11 GB

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
    dataDF;
  }

  def joinDF(path: String,
             dataFrame: DataFrame,
             spark: SparkSession): DataFrame = {

    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .load(path)
      .join(dataFrame, Seq("cuid"), "inner")
    df
  }
}

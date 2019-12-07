import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{array, explode, udf}

object Classification {

  val spark: SparkSession = SparkSession.builder().appName("Classifier")
    .config("spark.driver.maxResultSize", "1g")
    .config("spark.master", "local").getOrCreate()

  def main(args: Array[String]): Unit = {

    val testPath = "./mlboot_test.tsv" // 6MB
    val trainPath = "./mlboot_train_answers.tsv" // 15 MB


    val dataDF = loadDF()

    val testDf = joinDF(testPath, dataDF).show(4, truncate = false)
    val trainDf = joinDF(trainPath, dataDF)


    spark.stop()
  }

  def loadDF(): DataFrame = {
    val dataPath = "./mlboot_data.tsv" // 11 GB

    import spark.implicits._
    import org.apache.spark.sql.types._

    val schema = StructType(Array(
      StructField("cuid", StringType, nullable = true),
      StructField("cat_feature", IntegerType, nullable = true),
      StructField("feature_1", StringType, nullable = true),
      StructField("feature_2", StringType, nullable = true),
      StructField("feature_3", StringType, nullable = true),
      StructField("dt_diff", LongType, nullable = true))
    )

    val tempDataDF = spark.read.format("csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .schema(schema)
      .csv(spark.sparkContext.textFile(dataPath, 500).toDS())

    val dataDF = tempDataDF
      .withColumn("features", array(tempDataDF("feature_1"), tempDataDF("feature_2"), tempDataDF("feature_3")))
      .withColumn("features", explode($"features"))
      .withColumn("features", splitStringArray($"features"))
      .drop("feature_1")
      .drop("feature_2")
      .drop("feature_3")

    dataDF
  }

  def splitStringArray: UserDefinedFunction = udf((json: String) => {
    json.substring(1, json.length - 1).split(",").map(string => {
      string.trim.replace("\"", "")
    })
  })


  def joinDF(path: String,
             dataFrame: DataFrame): DataFrame = {

    val df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .load(path)
      .join(dataFrame, Seq("cuid"), "inner")

    df.printSchema()

    df
  }
}

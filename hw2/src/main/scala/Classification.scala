import org.apache.spark.ml.feature.{OneHotEncoderEstimator, VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{array, col, explode, udf}

object Classification {

  val spark: SparkSession = SparkSession.builder().appName("Classifier")
    .config("spark.driver.maxResultSize", "1g")
    .config("spark.master", "local").getOrCreate()

  def main(args: Array[String]): Unit = {

    val testPath = "./mlboot_test.tsv" // 6MB
    val trainPath = "./mlboot_train_answers.tsv" // 15 MB


    val dataDF = loadDF()

    val testDf = joinDF(testPath, dataDF).show(1000, truncate = false)
    val trainDf = joinDF(trainPath, dataDF).show(1000, truncate = false)

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

    val dataDF = spark.read.format("csv")
      .option("header", "false")
      .option("delimiter", "\t")
      .schema(schema)
      .csv(spark.sparkContext.textFile(dataPath, 500).toDS())

    dataDF
  }

  def vectorSparse: UserDefinedFunction = udf((json: String) => {
    val map: collection.mutable.Map[Int, Double] = collection.mutable.Map()
    json.substring(1, json.length - 1).split(",").map(_.trim.replace("\"", ""))
      .foreach(string => {
        if (!string.equals("")) {
          val splitStr = string.split(":")
          val index = splitStr(0).toInt
          val value = splitStr(1).toDouble
          map(index) = value
        }
      })
    var size = 0
    if (map.nonEmpty) {
      size = map.keysIterator.reduceLeft((x, y) => if (x > y) x else y) + 1
    }
    Vectors.sparse(size, map.toSeq)
  })

  def joinDF(path: String,
             dataFrame: DataFrame): DataFrame = {

    val tempDataDF = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", "\t")
      .load(path)
      .join(dataFrame, Seq("cuid"), "inner")

    val dataDF = tempDataDF
      .withColumn("features_j", array(tempDataDF("feature_1"), tempDataDF("feature_2"), tempDataDF("feature_3")))
      .withColumn("features_j", explode(col("features_j")))
      .withColumn("features_j", vectorSparse(col("features_j")))
      .drop("feature_1")
      .drop("feature_2")
      .drop("feature_3")

    val df = new OneHotEncoderEstimator()
      .setInputCols(Array("cat_feature"))
      .setOutputCols(Array("cat_vector"))
      .fit(dataDF)
      .transform(dataDF)
      .drop("cat_feature")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("dt_diff", "features_j", "cat_vector"))
      .setOutputCol("features")

    val asmDF = vectorAssembler.transform(df)
      .drop("features_j")
      .drop("cat_vector")
      .drop("dt_diff")

    asmDF.printSchema()
    asmDF
  }
}

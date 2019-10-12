import org.apache.spark.sql.{SparkSession, functions}
import org.apache.spark.SparkContext
import org.apache.spark.sql.types.{StructField, StructType, StringType}


object Application extends App {

  val spark = SparkSession.builder.appName("Ukraine").config("spark.master", "local").getOrCreate()

  val pathCSV = "./ua_reactions.csv"
  val pathTxt = "./excluded.txt"

  val csvDataFrame = spark.read.format("csv").option("header", "true").load(pathCSV) // header = true

  // exclude UAs

  val excludedFileDataFrame = spark.createDataFrame(
    spark.read.text(pathTxt).rdd,
    new StructType().add(StructField("ua", StringType, true)));

  // First task

  val csvDataFrameWithoutExcluded = csvDataFrame
    .join(excludedFileDataFrame, Seq("ua"), "left_anti")


  csvDataFrameWithoutExcluded.groupBy("ua") // group by ua
    .agg(functions.sum("is_click").alias("clicks"), functions.count("is_click").alias("shows")) //  sum clicks for each ua
    .where("clicks > 5")
    .orderBy(functions.desc("clicks"))
    .withColumn("CTR", functions.col("clicks") / functions.col("shows"))
    .show(5) // show 5



  // Second task

  val total = csvDataFrame.agg(functions.sum("is_click")).first.getDouble(0).toInt

  val PercentsDF = csvDataFrame.groupBy("ua") // group by ua
    .agg(functions.sum("is_click").alias("clicks"))
    .withColumn("percents", functions.col("clicks") / functions.sum("clicks").over())

  println(PercentsDF
    .orderBy(functions.desc("percents"))
    .limit(14)
    .agg(functions.sum("percents"))
    .first.getDouble(0))
}

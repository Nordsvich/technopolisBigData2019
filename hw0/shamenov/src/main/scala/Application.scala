import org.apache.spark.sql.{ SparkSession, functions}
import org.apache.spark.sql.types.{StringType, StructField, StructType}


object Application extends App {

  val spark = SparkSession.builder.appName("Ukraine").config("spark.master", "local").getOrCreate()

  val pathCSV = "./ua_reactions.csv"
  val pathTxt = "./excluded.txt"

  var csvDataFrame = spark.read.format("csv").option("header", "true").load(pathCSV) // header = true

  // exclude UAs

  val excludedFileDataFrame = spark.createDataFrame(
    spark.read.text(pathTxt).rdd,
    new StructType().add(StructField("ua", StringType, true)));

  // First task

  csvDataFrame = csvDataFrame
    .join(excludedFileDataFrame, Seq("ua"), "left_anti")

  csvDataFrame = csvDataFrame.groupBy("ua") // group by ua
    .agg(functions.sum("is_click").alias("clicks"), functions.count("is_click").alias("shows")) //  sum clicks for each ua
    .where("clicks > 5")
    .orderBy(functions.desc("clicks"))
    .withColumn("CTR", functions.col("clicks") / functions.col("shows")).toDF()
    //.show(5) // show 5

  csvDataFrame.show(5)

  // Second task

  val shows = csvDataFrame
    .agg(functions.sum("shows"))
    .first
    .getLong(0).toInt

  val total = shows / 2

  println("shows:" + shows)
  println("total is : " + total)

  var sum : Long = 0
  var index: Int = 0

  csvDataFrame
    .collect().foreach(row => {
        sum = sum + row(2).toString.toLong
        if(sum <= total)  {
          index = index +1
        }
  })

  println("index : " + index)

  csvDataFrame
    .show(index)
}

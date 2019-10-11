import org.apache.spark.sql.{SparkSession, functions}
import org.apache.spark.SparkContext

object Application extends App {

  val spark = SparkSession.builder.appName("Ukraine").config("spark.master", "local").getOrCreate()
  val path = "./ua_reactions.csv"
  val textFileDataFrame = spark.read.format("csv").option("header", "true").load(path) // header = true


  // exclude UAs

  val excludedFileDataFrame = spark.read.format("txt") // load txt

  // First task

  textFileDataFrame.groupBy("ua") // group by ua
                              .agg(functions.sum("is_click").alias("clicks"), functions.count("is_click").alias("shows")) //  sum clicks for each ua
                              .where("clicks > 5") // where > 5 clicks
                              .orderBy(functions.desc("clicks")) // sort by desc
                              .withColumn("CTR", functions.col("clicks") / functions.col("shows"))
                              .show(5) // show 5

  // Second task

  



  val total = textFileDataFrame.agg(functions.sum("is_click")).first.getDouble(0).toInt //

  val PercentsDF = textFileDataFrame.groupBy("ua") // group by ua
            .agg(functions.sum("is_click").alias("clicks"))
            .withColumn("percents", functions.col("clicks") / functions.sum("clicks").over()) // + new column

  //println(PercentsDF.orderBy(functions.desc("percents")).limit(14).agg(functions.sum("percents")).first.getDouble(0))

}

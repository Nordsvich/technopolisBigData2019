import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window.orderBy

object HomeWork {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("HomeWork")
      .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")
    val df = spark.read.format("csv")
      .option("header", "true")
      .load("../ua_reactions.csv")
    //df.describe().show()
    val excluded = spark.read.format("text")
      .load("../excluded.txt")
    //excluded.describe().show()

    val filteredDF = df
      .select("*")
      .join(excluded, df("ua") === excluded("value"), "leftanti")
    //filteredDF.describe().show()

    val groupedDF = filteredDF
      .groupBy("ua")
      .agg(count("*").alias("shows"), sum("is_click").alias("clicks"))
      .withColumn("CTR", expr("clicks/shows"))

    val CTR_top_5 = groupedDF
      .filter(col("shows") >= 5)
      .orderBy(desc("CTR"))

    CTR_top_5.show(5, false)

    val sum_shows = groupedDF
      .agg(sum("shows"))
      .first()
      .getLong(0);

    val half_shows = groupedDF
      .withColumn("sum_shows", sum("shows").over(orderBy(desc("shows"))))
      .filter(col("sum_shows") <= sum_shows / 2)

    half_shows.show(false)

    spark.stop()

  }
}

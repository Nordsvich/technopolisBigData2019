import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window.orderBy
import org.apache.spark.sql.functions._

object UserAgentStats {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local")
      .appName("UserAgentStats")
      .getOrCreate()
    spark.sparkContext.setLogLevel("OFF")

    val df = spark.read.format("csv")
      .option("header", "true")
      .load(getClass.getResource("ua_reactions.csv").getPath)

    val excluded = spark.read.format("text")
      .load(getClass.getResource("excluded.txt").getPath)

    println("Before filter: " + df.count())

    val df_filtered = df.join(excluded,
      df.col("ua") === excluded.col("value"),
      "leftanti")

    println("After filter: " + df_filtered.count())

    val df_counted = df_filtered.groupBy(col("ua"))
      .agg(
        count("*").alias("shows"),
        sum(col("is_click")).alias("clicks"))
      .filter(col("shows") > 5)
      .withColumn("CTR", expr("clicks/shows"))
      .orderBy(col("CTR").desc)
      .limit(5)

    df_counted.show(false)

    val df_moreThanHalf = df_filtered.groupBy(col("ua"))
      .agg(
        count("*").alias("shows"),
        sum(col("is_click")).alias("clicks"))
      .withColumn("shows_sum",
        sum("shows")
          .over(orderBy(col("shows")))
      )
      .filter(col("shows_sum") >= (df_filtered.count() / 2))
      .drop(col("shows_sum"))

    df_moreThanHalf.show(false)

    spark.stop()

  }
}

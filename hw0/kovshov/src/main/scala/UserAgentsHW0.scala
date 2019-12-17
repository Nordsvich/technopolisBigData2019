import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

object UserAgentsHW0 {

  val PATH_TO_REACTIONS_CSV: String = "./ua_reactions.csv"
  val PATH_TO_EXCLUDED_TXT: String = "./excluded.txt"

  def main(args: Array[String]): Unit = {
    val sparkSession = createSparkSession()

    val reactions_df = sparkSession.read
      .format("csv")
      .option("header", value = true)
      .load(PATH_TO_REACTIONS_CSV)

    val excluded_df = sparkSession.read
      .format("text")
      .load(PATH_TO_EXCLUDED_TXT)

    val grouped_df = reactions_df
      .select("*")
      .join(
        broadcast(excluded_df),
        reactions_df("ua") === excluded_df("value"),
        "left_anti")
      .groupBy(col("ua"))
      .agg(
        count("*").alias("shows"),
        sum(col("is_click")).alias("clicks"))
      .withColumn(
        "CTR",
        round(expr(String.format("%s / %s", "clicks", "shows")), 3))

    val first_task_df = grouped_df
      .filter(col("shows") > 5)
      .orderBy(desc("CTR"))
      .limit(5)

    println("First task")
    first_task_df.show()

    val sumShows = grouped_df
      .agg(sum("shows"))
      .first()
      .getLong(0)

    val second_task_df = grouped_df
      .withColumn(
        "percentile",
        col("shows") / sumShows * 100)
      .withColumn(
        "total_percentile",
        round(sum("percentile").over(Window.orderBy(desc("percentile"))), 3))
      .filter(col("total_percentile") <= 50)
      .drop(col("percentile"))
      .drop(col("CTR"))

    println("Second task")
    second_task_df.show()

  }

  def createSparkSession(): SparkSession = {
    val spark = SparkSession.builder
      .appName("User Agents")
      .master("local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("OFF")
    spark
  }
}

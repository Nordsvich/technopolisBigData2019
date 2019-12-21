import org.apache.spark.sql.expressions.Window.orderBy
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object UserAgents {
  val uaReactionsPath = "../ua_reactions.csv"
  val uaExcludedPath = "../excluded.txt"

  def main(args: Array[String]) {
    val spark = createSparkSession()
    val uaReactions = spark.read.option("header", "true").csv(uaReactionsPath).cache()
    val uaExcluded = spark.read.text(uaExcludedPath).toDF("ua").cache()

    val uaSummary = uaReactions
      .join(uaExcluded, Seq("ua"), "left_anti")
      .groupBy("ua")
      .agg(
        count(lit(1)).alias("shows"),
        sum(col("is_click")).alias("clicks")
      ).cache()

    topByCtr(uaSummary, spark)
    halfShows(uaSummary)

    spark.stop()
  }

  /**
   * Find top 5 User Agents by CTR where shows > 5 and CTR = clicks / shows
   */
  def topByCtr(uaSummary: DataFrame, spark: SparkSession): Unit = {
    val uaSummaryCtr = uaSummary
      .filter(col("shows") > 5)
      .withColumn("ctr", col("clicks") / col("shows"))
    val uaTopCtr = uaSummaryCtr
      .sort(desc("ctr"))
      .takeAsList(5)
    println("Top 5 user agents by CTR:")
    spark.createDataFrame(uaTopCtr, uaSummaryCtr.schema).show(false)
  }

  /**
   * Find User Agents that make up half of all views
   */
  def halfShows(uaSummary: DataFrame): Unit = {
    val totalShows = colSum(uaSummary, "shows")
    val uaHalfShows = uaSummary
      .withColumn("cum_shows", sum("shows").over(orderBy(desc("shows"))))
      .filter(col("cum_shows") <= totalShows / 2)
    val halfShows = colSum(uaHalfShows, "shows")
    println(s"Agents with half shows ($halfShows out of $totalShows): ")
    uaHalfShows.show(false)
  }

  def createSparkSession(): SparkSession = {
    val spark = SparkSession.builder
      .appName("User Agents")
      .master("local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark
  }

  def colSum(df: DataFrame, col: String): Long = df.agg(sum(col)).first.getLong(0)
}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

object UAConverter {

  val CSV_PATH: String = "../ua_reactions.csv"
  val EXCL_PATH: String = "../excluded.txt"

  val UA_COL: String = "ua"
  val SHOWS_COL: String = "shows"
  val CLICKS_COL: String = "clicks"
  val ALL: String = "*"
  val CTR_COL: String = "CTR"
  val PERC_COL: String = "percentile"
  val TOTAL_PERC_COL: String = "total_percentile"

  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder().master("local")
      .appName("UAConverter").getOrCreate()
    //Just for output
    sparkSession.sparkContext.setLogLevel("WARN")

    val data = sparkSession.read.format("csv")
      .option("header", value = true).load(CSV_PATH)

    val excluded = sparkSession.read.format("text")
      .load(EXCL_PATH)

    val filter = data.select(ALL).join(broadcast(excluded), data(UA_COL) === excluded("value"),
      "left_anti").repartition(1)

    val groupedData = filter.groupBy(col(UA_COL))
      .agg(count(ALL).alias(SHOWS_COL), sum(col("is_click"))
        .alias(CLICKS_COL)).withColumn(CTR_COL, round(expr(String.format("%s / %s", CLICKS_COL, SHOWS_COL)), 2))

    val moreThanFive = groupedData.filter(col(SHOWS_COL) > 5)

    val topFive = moreThanFive.orderBy(desc(CTR_COL)).limit(5)
    println("Task 1: ")
    topFive.show()

    val sumShows = groupedData.agg(sum(SHOWS_COL)).first().getLong(0)

    val half = groupedData.withColumn(PERC_COL, col(SHOWS_COL) / sumShows * 100)
      .withColumn(TOTAL_PERC_COL, round(sum(PERC_COL).over(Window.orderBy(desc(PERC_COL))), 2))
      .filter(col(TOTAL_PERC_COL) <= 50).drop(col(PERC_COL)).drop(col(CTR_COL))
    println("Task 2: ")
    half.show()

  }
}

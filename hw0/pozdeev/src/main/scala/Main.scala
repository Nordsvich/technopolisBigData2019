import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

object Main extends App {
  val spark = SparkSession.builder
    .master("local[*]")
    .appName("BigData2019")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  val reactions = spark.read
    .option("header", true)
    .csv("../ua_reactions.csv")

  val excluded = spark.read.text("../excluded.txt")

  val filteredReactions = reactions.as("reac")
    .join(excluded.as("ex"), $"reac.ua" === $"ex.value", "leftanti")

  println("1) Взять все UA для которых было больше 5 показов рекламы посчитать CTR (clicks / shows) для каждого UA\nи вывести топ 5.")

  filteredReactions
    .groupBy($"ua")
    .agg(sum($"is_click").as("clicks"), count($"is_click").as("shows"))
    .filter($"shows" > 5)
    .withColumn("CTR", $"clicks" / $"shows")
    .sort($"CTR".desc)
    .show(5)

  println("2) Вывести все UA на которых приходится 50% рекламных показов.")

  val totalShows = filteredReactions.count()

  filteredReactions
    .groupBy($"ua")
    .agg(count($"is_click").as("shows"))
    .withColumn("%", $"shows" / totalShows * 100)
    .withColumn("%_sum", sum($"%").over(Window.orderBy($"%".desc)))
    .filter($"%_sum" < 50)
    .show()

}
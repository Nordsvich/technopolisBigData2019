package sample

import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType}

object Main extends App {

  override def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("User Agents")
      .master("local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    println("LOAD SPARK")

    val dfReactions: DataFrame = spark.read.option("header", "true").csv("hw0/ua_reactions.csv").cache()
    val dfExcluded: DataFrame = spark.read.text("hw0/excluded.txt").toDF("ua").cache()

    dfExcluded.printSchema()
    dfReactions.printSchema()

    val arrayExcluded = dfExcluded.collect()

    val dfMain = dfReactions
      .filter(r => !contains(r(0), arrayExcluded))
      .groupBy("ua")
      .agg(sum("is_click"), count("is_click"))


    dfMain.withColumn("ctr", dfMain.col(dfMain.columns(1)) / dfMain.col(dfMain.columns(2)))
      .filter(col(dfMain.columns(2)).>(5))
      .orderBy(desc("ctr"))
      .show(5, truncate = false)


  }


  def getShows(userAgent: Any, array: Array[Row]): Any = array.find(r => r(0) == userAgent)

  def contains(userAgent: Any, array: Array[Row]): Boolean = array.contains(userAgent)
}

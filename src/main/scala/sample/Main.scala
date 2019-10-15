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

    val arrayExcluded = dfExcluded.collect()

    val dfMain = dfReactions
      .filter(r => !contains(r(0), arrayExcluded))
      .groupBy("ua")
      .agg(sum("is_click").alias("clicked"), count("is_click").alias("showed"))


    //Task 1

    dfMain.withColumn("ctr", dfMain.col(dfMain.columns(1)) / dfMain.col(dfMain.columns(2)))
      .filter(col(dfMain.columns(2)).>(5))
      .orderBy(desc("ctr"))
      .show(5, truncate = false)


    //Task 2

    val arrayMain = dfMain.orderBy(desc(dfMain.columns(2))).collect()
    var totalShows: Long = 0

    arrayMain.foreach(r => {
      totalShows += r(2).asInstanceOf[Long]
    })

    var countAns: Int = 0
    var countShows: Long = 0
    arrayMain.foreach(r => {
      if (countShows + r(2).asInstanceOf[Long] < totalShows / 2) {
        countShows += r(2).asInstanceOf[Long]
        countAns = countAns + 1
      }
    })

    dfMain.orderBy(desc(dfMain.columns(2))).show(countAns, truncate = false)
  }


  def getShows(userAgent: Any, array: Array[Row]): Any = array.find(r => r(0) == userAgent)

  def contains(userAgent: Any, array: Array[Row]): Boolean = array.contains(userAgent)
}

package com.github.senyast4745.firstML

import org.apache.spark.sql.{DataFrame, SparkSession}

object MainClass {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("Spark SQL basic example")
    .config("spark.master", "local")
    .getOrCreate()

  val df: DataFrame = spark.read.csv("/home/arseny/WorkingFolder/TechoPolis/ML/FirstMLTask/hw0/ua_reactions.csv")
  val df1: DataFrame = spark.read.text("/home/arseny/WorkingFolder/TechoPolis/ML/FirstMLTask/hw0/excluded.txt")

  def main(args: Array[String]): Unit = {
   /* df1.show()
    df.show()
    df.printSchema()*/
    print(df.collect().length)
    val list = df1.collect()
    val answ = df.filter(r => r(1) == "1" && !list.contains(r(1)))
    answ.show()
    print(answ.collect().length)
  }
}

package com.github.senyast4745.firstML

import org.apache.spark.sql.{DataFrame, SparkSession, DataFrameNaFunctions}
import  org.apache.spark.sql.functions;


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
//    val answ = df.filter(r => r(1) == "1" && !list.contains(r(0))).describe()
//    answ.show()
//df.filter(r => !list.contains(r(0))).groupBy(df.columns.apply(0).select).show()
      // ...copy paste that for columns y and z

    val as =  df.filter(r => r(1) == "1" && !list.contains(r(0))).groupBy(df.columns.apply(0))
    .count()
    as.sort(as.col("count").desc).show(5)
    /*.where(functions.col('count') > 1)
    .select(functions.sum('count'))
    .show()*/
//    print(answ.collect().length)
  }
}

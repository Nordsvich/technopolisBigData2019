package com.github.senyast4745.firstML

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

object MainClass {

	def main(args: Array[String]): Unit = {
		val spark =  SparkSession.builder
			.appName("User Agents")
			.master("local")
			.getOrCreate()

		val mainDf = spark.read.option("header", "true").csv("./hw0/ua_reactions.csv").toDF().cache()
		val exDf = spark.read.text("./hw0/excluded.txt").toDF("value").cache()

		val groupDf = mainDf
			.select("*")
			.join(
				broadcast(exDf),
				mainDf("ua") === exDf("value"),
				"left_anti")
			.groupBy(col("ua"))
			.agg(
				count("*").alias("shows"),
				sum(col("is_click")).alias("clicks"))
			.withColumn(
				"CTR",
				round(expr(String.format("%s / %s", "clicks", "shows")), 3))

		val first_task_df = groupDf
			.filter(col("shows") > 5)
			.orderBy(desc("CTR"))
			.limit(5)

		first_task_df.show()

		val sumShows = groupDf
			.agg(sum("shows"))
			.first()
			.getLong(0)

		val second_task_df = groupDf
			.withColumn(
				"per",
				col("shows") / sumShows * 100)
			.withColumn(
				"total_percentile",
				round(sum("per").over(Window.orderBy(desc("per"))), 3))
			.filter(col("total_percentile") <= 50)

		second_task_df.show()

	}
}


import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object UaWorker {

  def main(args: Array[String]): Unit = {
    //Стартуем спарк и достаём наши данные
    val spark: SparkSession = SparkSession.builder().master("local").appName("hw0").getOrCreate()
    val data: DataFrame = spark.read.format("csv").option("header", "true").load("hw0/ua_reactions.csv")
    val dataToExclude: DataFrame = spark.read.format("text").load("hw0/excluded.txt")

    //Исключаем то что нужно исключить
    val dataExcluded = data.join(dataToExclude, col("ua") === col("value"), "left_anti")

    //Группируем и считаем кол-во показов и кол-во кликов
    val dataGrouped = dataExcluded
      .groupBy("ua")
      .agg(
        //Кол-во показов всего
        count("*").alias("shows"),
        //Клики считаем только когда столбец "is_click" равен 1
        sum(when(col("is_click") === 1, 1).otherwise(0)).as("clicks")
      )

    //Отсеиваем меньше 6 показов
    val dataFiltered = dataGrouped.filter(col("shows") > 5)
    //Считаем CTR
    val dataCountedCTR = dataFiltered.withColumn("CTR", expr("clicks/shows"))
    //Сортируем
    val dataOrdered = dataCountedCTR.orderBy(desc("CTR"))

    //Получаем наш первый ответ
    dataOrdered.show(5, truncate = false)

    //Считаем общее кол-во просмотров
    val totalViews = dataExcluded.count()
    //Выражение по которому будем считать процент просмотров
    val expression = "shows / %d".format(totalViews)

    //Выводим все UA на которых приходится 50% рекламных показов.
    val dataShowPercentFiltered = dataGrouped.withColumn("showPercent", expr(expression)).filter(col("showPercent") === 0.5)
    //Выводим наш ответ
    dataShowPercentFiltered.show(truncate = false)

    //Стопим спарк
    spark.stop()
  }

}

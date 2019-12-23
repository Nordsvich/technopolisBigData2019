import org.apache.spark.sql.types.{StructField, StructType, StringType}
import org.apache.spark.sql.{functions,SparkSession}
import org.apache.spark.sql.functions._

object MyObj {

   def main(args: Array[String]) {

    val spark = SparkSession.builder.master("local[*]").appName("nadenokk")
      .config("spark.executor.cores","4").config("spark.executor.memory","4G").getOrCreate()
    spark.sparkContext.setLogLevel("OFF")

    val  exld = spark.createDataFrame(spark.read.text("../excluded.txt").rdd, new StructType().add(StructField("ua",StringType,true)))

    val cvs = spark.read.option("header", "true").csv("../ua_reactions.csv")

    val followersDF=cvs.join(exld,Seq("ua"),"left_anti")

    val followingDF = cvs.groupBy("ua")
      .agg(count("ua").alias("is_clicks")).orderBy(desc("is_clicks"))

     followingDF.createOrReplaceTempView("r1")

    val followingClicksDF = cvs.filter("is_click=1").groupBy("ua")
      .agg(count("ua").alias("clicks")).orderBy(desc("clicks"))

      followingClicksDF.createOrReplaceTempView("r2")

    spark.sql("select r2.ua,r2.clicks/r1.is_clicks " +
      "as res from r1 inner join r2 on r1.ua=r2.ua where is_clicks > 5 order by res desc limit 5").show(10000,false)


     val sum  = followingDF.select("is_clicks")
       .agg(functions.sum("is_clicks")).rdd.take(1)(0).toSeq(0).toString.toInt/2
    var s: Int = 0
    var i: Int = 0
    followingDF.collect().foreach(r =>{
      s +=r(1).toString.toInt
      if (s < sum) i= i+1
    })

    followingDF.limit(i).show(100,false)

  }
}

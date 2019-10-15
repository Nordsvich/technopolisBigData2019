import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

object MyClass {
  def main(args: Array[String]) {
    //val sc=new SparkContext("local[*]", "elenapranova");
    val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("elenapranova")
    .getOrCreate()
    val excluded=spark.read.textFile("../excluded.txt")
    val reactions=spark.read.csv("../ua_reactions.csv")

    excluded.createGlobalTempView("excluded")
    reactions.createGlobalTempView("ua_reactions")

    val tableClick=spark.sql("SELECT _c0, count(_c0) as click FROM global_temp.ua_reactions where _c1=1 group by _c0")
    val tableShow=spark.sql("SELECT _c0, count(_c0) as show FROM global_temp.ua_reactions group by _c0")

    tableClick.createGlobalTempView("tableForResult1")
    tableShow.createGlobalTempView("tableForResult2")

    val pre_result=spark.sql("select global_temp.tableForResult1._c0, click/show as res FROM global_temp.tableForResult1 inner join global_temp.tableForResult2 on global_temp.tableForResult1._c0=global_temp.tableForResult2._c0 where click>=5 order by res desc limit 5")

    pre_result.createGlobalTempView("Result")

    val result=spark.sql("select * from global_temp.Result where _c0 not in (select value from global_temp.excluded)")

    result.show(false) //done first task

    val countAllShows = spark.sql("select count(*) as c from global_temp.ua_reactions where _c0 not in (select value from global_temp.excluded)")//544


    val pre_res=spark.sql("select global_temp.tableForResult1._c0 as _c0, click+show as sum FROM global_temp.tableForResult1 inner join global_temp.tableForResult2 on global_temp.tableForResult1._c0=global_temp.tableForResult2._c0")
    
    pre_res.createGlobalTempView("pre_res")

    val pre_result2=spark.sql("select * from global_temp.pre_res where _c0 not in (select value from global_temp.excluded)")

    pre_result2.createGlobalTempView("pre_result2")

    val result2=spark.sql("select _c0, sum/544 as rew from  global_temp.pre_result2")

    result2.createGlobalTempView("result2")

    val resultTask2=spark.sql("select _c0, rew from global_temp.result2 where rew>=0.05 and rew<0.06")

    resultTask2.show(false)//done second task
  }
}

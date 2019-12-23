import org.apache.spark.ml.classification.{ RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.functions.{avg, col}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

object main {

def main(args: Array[String]): Unit = {
  val spark = SparkSession.builder()
    .master("local")
    .appName("hw2")
    .config("spark.executor.cores", "4")
    .config("spark.executor.memory", "4G")
    .getOrCreate()
  spark.sparkContext.setLogLevel("OFF")

  val dfTrain = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("./train.csv")
    .withColumn("label", col("Survived"))

  val dfTrainPreproc = preprocess(dfTrain)
    .select("features", "label")
    .cache()
  val Array(trainData, testData) = dfTrainPreproc.randomSplit(Array(0.8, 0.2))

  val randomForest = new RandomForestClassifier()
  val paramGrid = new ParamGridBuilder()
    .addGrid(randomForest.maxDepth, Array(5, 10, 15))
    .addGrid(randomForest.maxBins, Array(30, 40, 50))
    .addGrid(randomForest.impurity, Array("gini", "entropy"))
    .build()
  // cross validate
  val cv = new CrossValidator()
    .setEstimator(randomForest)
    .setEvaluator(new BinaryClassificationEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(5)
    .setParallelism(2)
  // fit model
  val model = cv.fit(trainData)
  val bestRandomForest = model.bestModel.asInstanceOf[RandomForestClassificationModel]

  // evaluate
  val predictions = bestRandomForest.transform(testData)
  val evaluator = new BinaryClassificationEvaluator()
    .setRawPredictionCol("prediction")
    .setLabelCol("label")

  for (metric <- Seq("areaUnderROC", "areaUnderPR")) {
    println(s"$metric = ${evaluator.setMetricName(metric).evaluate(predictions)}")
  }

  val dfTest = spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("test.csv")

  val dfTestPreproc = preprocess(dfTest)
    .select("PassengerId", "features")
    .cache()

  bestRandomForest.transform(dfTestPreproc)
    .select("PassengerId", "prediction")
    .withColumn("Survived", col("prediction").cast(IntegerType))
    .drop("prediction")
    .write
    .option("header", "true")
    .mode(SaveMode.Overwrite)
    .csv("./predictions")
}

  def preprocess(dataFrame: DataFrame): DataFrame = {

    val  avgAge: Double  = dataFrame.select(col = "Age")
      .agg(avg(columnName = "Age"))
      .collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }
    val avgFare:Double = dataFrame.select(col = "Fare")
      .agg(avg(columnName = "Fare"))
      .collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }

    val dfFixed = dataFrame.na.fill(Map(
      "Age" -> avgAge,
      "Fare" -> avgFare,
      "Embarked"-> "S"
    ))

    val sexProc= stringEncoder("Sex")
    val embarkedProc = stringEncoder("Embarked")

    val features = Array("Age", "Fare", "SexOneHot", "EmbarkedOneHot", "Pclass", "SibSp", "Parch")
    val assembler = new VectorAssembler()
    assembler.setInputCols(features).setOutputCol("features")

    val pipeline = new Pipeline().setStages(sexProc ++ embarkedProc  ++ Array(assembler))
    pipeline.fit(dfFixed)
      .transform(dfFixed)
  }

  def stringEncoder(colName : String): Array[PipelineStage] = {

    val stringIndexer = new StringIndexer().setInputCol(colName)
      .setOutputCol(colName+"Indexed")
      .setHandleInvalid("skip")

    val oneHotEncoderEstimator = new OneHotEncoderEstimator().setInputCols(Array(colName+"Indexed"))
      .setOutputCols(Array(colName+"OneHot"))

    Array(stringIndexer, oneHotEncoderEstimator)
  }

}
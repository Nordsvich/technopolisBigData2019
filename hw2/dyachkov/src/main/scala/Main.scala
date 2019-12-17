import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

object Main {
  val trainPath = "../train.csv"
  val testPath = "../test.csv"
  val predictionsPath = "../predictions.csv"

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = createSparkSession()
    val model = createModel(spark, trainPath)
    makePredictions(spark, model, testPath, predictionsPath)
  }

  def createModel(spark: SparkSession, path: String): PipelineModel = {
    // read
    var df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)
      .drop("Cabin") // too many nulls
      .drop("Ticket") // useless
      .withColumn("label", col("Survived"))

    // handle missing values
    df = df.na
      .fill(Map(
        ("Age", getAvg(df, "Age")), // average age
        ("Embarked", "S"))) // only 2 nulls, whatever

    // split
    val Array(train, test) = df.randomSplit(Array(0.7, 0.3), seed = 42)

    // features
    val catFeaturesNames = Array("Pclass", "Sex", "Embarked")
    val catFeaturesIndexed = catFeaturesNames.map(_ + "Indexed")
    val numFeatures = Array("Age", "SibSp", "Parch", "Fare")
    val features = numFeatures ++ catFeaturesIndexed

    // model
    val indexers = catFeaturesNames.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "Indexed")
    }

    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val randomForest = new RandomForestClassifier()

    val pipeline = new Pipeline()
      .setStages(indexers ++ Array(assembler, randomForest))

    val paramGrid = new ParamGridBuilder()
      .addGrid(randomForest.maxDepth, Array(5, 10, 15))
      .addGrid(randomForest.maxBins, Array(10, 20, 30))
      .addGrid(randomForest.impurity, Array("gini", "entropy"))
      .build()

    // cross validation
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(2)

    val model = cv.fit(train)
    val bestModel = model.bestModel.asInstanceOf[PipelineModel]
    val bestRandomForest = bestModel.stages(4).asInstanceOf[RandomForestClassificationModel]

    println("RandomForest hyperparameters: " +
      s"maxDepth = ${bestRandomForest.getMaxDepth}, " +
      s"maxBins = ${bestRandomForest.getMaxBins}, " +
      s"impurity = ${bestRandomForest.getImpurity}")

    // evaluation
    val predictions = bestModel.transform(test)
    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol("prediction")
      .setLabelCol("label")

    for (metric <- Seq("areaUnderROC", "areaUnderPR")) {
      println(s"$metric = ${evaluator.setMetricName(metric).evaluate(predictions)}")
    }

    bestModel
  }

  def makePredictions(spark: SparkSession, model: PipelineModel, testPath: String, predictionPath: String): Unit = {
    // read
    var df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(testPath)
    
    // handle missing values
    df = df.na
      .fill(Map(
        ("Fare", getAvg(df, "Fare")),
        ("Age", getAvg(df, "Age"))))

    val predictions = model.transform(df)
    
    predictions.select("PassengerId", "prediction")
      .withColumn("Survived", predictions.col("prediction").cast(IntegerType))
      .drop("prediction")
      .write
      .option("header", "true")
      .mode(SaveMode.Overwrite)
      .csv(predictionPath)
  }

  def createSparkSession(): SparkSession = {
    val spark = SparkSession.builder
      .appName("Spark ML")
      .master("local")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark
  }

  def getAvg(df: DataFrame, colName: String): Double = {
    df.select(colName)
      .agg(avg(colName))
      .collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }
  }
}

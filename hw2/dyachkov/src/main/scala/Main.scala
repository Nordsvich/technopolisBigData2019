import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Model, Pipeline}
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

  def createModel(spark: SparkSession, path: String): Model[_] = {
    // read
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)
      .withColumn("label", col("Survived"))

    // preprocess
    val processed = preprocess(df)
      .select("features", "label")
      .cache()

    // split
    val Array(train, test) = processed.randomSplit(Array(0.7, 0.3), seed = 42)

    // model
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
    val model = cv.fit(train)
    val bestRandomForest = model.bestModel.asInstanceOf[RandomForestClassificationModel]

    println("RandomForest hyperparameters: " +
      s"maxDepth = ${bestRandomForest.getMaxDepth}, " +
      s"maxBins = ${bestRandomForest.getMaxBins}, " +
      s"impurity = ${bestRandomForest.getImpurity}")

    // evaluate
    val predictions = bestRandomForest.transform(test)
    val evaluator = new BinaryClassificationEvaluator()
      .setRawPredictionCol("prediction")
      .setLabelCol("label")

    for (metric <- Seq("areaUnderROC", "areaUnderPR")) {
      println(s"$metric = ${evaluator.setMetricName(metric).evaluate(predictions)}")
    }

    bestRandomForest
  }

  def makePredictions(spark: SparkSession, model: Model[_], testPath: String, predictionPath: String): Unit = {
    // read
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(testPath)

    // preprocess
    val processed = preprocess(df)
      .select("PassengerId", "features")
      .cache()

    // predict and save
    model.transform(processed)
      .select("PassengerId", "prediction")
      .withColumn("Survived", col("prediction").cast(IntegerType))
      .drop("prediction")
      .write
      .option("header", "true")
      .mode(SaveMode.Overwrite)
      .csv(predictionPath)
  }

  def preprocess(data: DataFrame): DataFrame = {
    // handle missing values
    val df = data.na
      .fill(Map(
        ("Age", getAvg(data, "Age")), // average age
        ("Embarked", "S"), // only 2 nulls, whatever
        ("Fare", getAvg(data, "Fare")))) // only 1 null, whatever

    // features
    val catFeaturesNames = Array("Pclass", "Sex", "Embarked")
    val catFeaturesIndexed = catFeaturesNames.map(_ + "Indexed")
    val catFeaturesOneHot = catFeaturesNames.map(_ + "OneHot")
    val numFeatures = Array("Age", "SibSp", "Parch", "Fare")
    val features = numFeatures ++ catFeaturesOneHot

    // index categorical
    val indexers = catFeaturesNames.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "Indexed")
    }

    // one hot from indexed categorical
    val oneHot = new OneHotEncoderEstimator()
      .setInputCols(catFeaturesIndexed)
      .setOutputCols(catFeaturesOneHot)

    val assembler = new VectorAssembler()
      .setInputCols(features)
      .setOutputCol("features")

    val pipeline = new Pipeline()
      .setStages(indexers ++ Array(oneHot, assembler))

    pipeline
      .fit(df)
      .transform(df)
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

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.functions.{avg, col, udf}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}

object Titanic {
  val trainPath = "./train.csv"
  val testPath = "./test.csv"
  val predictionPath = "./predictions"

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("HW2")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val gBTCModel = createGBTCModel(spark)

    val dfTest = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(testPath)

    val processed = preprocessDF(dfTest)
      .select("PassengerId", "features")
      .cache()

    gBTCModel.transform(processed)
      .select("PassengerId", "prediction")
      .withColumn("Survived", col("prediction").cast(IntegerType))
      .drop("prediction")
      .write
      .option("header", "true")
      .mode(SaveMode.Overwrite)
      .csv(predictionPath)
    //coalesce(1) doesn't work (or I don't know how to use it), so result will be partitioned
  }

  private def createGBTCModel(spark: SparkSession) = {
    val dfTrain = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(trainPath)
      .withColumn("label", col("Survived"))

    val preprocessed = preprocessDF(dfTrain)
      .select("features", "label")
      .cache()

    val Array(trainData, testData) = preprocessed.randomSplit(Array(0.7, 0.3))

    val gBTClassifier = new GBTClassifier()
    val paramGrid = new ParamGridBuilder()
      .addGrid(gBTClassifier.maxBins, Array(24, 32, 48))
      .addGrid(gBTClassifier.maxDepth, Array(5, 10, 15))
      .addGrid(gBTClassifier.maxIter, Array(5, 10, 20))
      .addGrid(gBTClassifier.impurity, Array("gini", "entropy"))
      .build()

    val crossValidator = new CrossValidator()
      .setEstimator(gBTClassifier)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(4)

    val model = crossValidator.fit(trainData)
    val gBTCBestModel = model.bestModel.asInstanceOf[GBTClassificationModel]

    gBTCBestModel
  }

  def preprocessDF(df: DataFrame): DataFrame = {
    //if we don't have data then replace it with average
    val dfFixedAgeAndFare = df.na.fill(Map(
      "Age" -> avgNumericColumn(df, "Age"),
      "Fare" -> avgNumericColumn(df, "Fare")
    ))

    val embarkedTrsf: (String => String) = {
      case "" => "S"
      case null => "S"
      case a => a
    }
    val embarkedUDF = udf(embarkedTrsf)

    val dfFixedAll = dfFixedAgeAndFare.withColumn("Embarked",
      embarkedUDF(dfFixedAgeAndFare.col("Embarked")))

    //process categorical features
    val sexProcessed = processCategoricalFeatures("Sex")
    val embarkedProcessed = processCategoricalFeatures("Embarked")
    val pClassProcessed = processCategoricalFeatures("Pclass")

    val features = Array("Age", "Fare", "SexOneHot", "EmbarkedOneHot", "PclassOneHot", "SibSp", "Parch") //others - useless

    val assembler = new VectorAssembler()
    assembler.setInputCols(features).setOutputCol("features")

    //create final pipeline with all changes
    val pipeline = new Pipeline().setStages(sexProcessed ++ embarkedProcessed ++ pClassProcessed ++ Array(assembler))
    pipeline.fit(dfFixedAll)
      .transform(dfFixedAll)
  }

  def avgNumericColumn(df: DataFrame, column: String): Double = {
    df.select(col = column)
      .agg(avg(columnName = column))
      .collect() match {
      case Array(Row(avg: Double)) => avg
      case _ => 0
    }
  }

  def processCategoricalFeatures(column: String): Array[PipelineStage] = {
    //first using stringindexer
    val stringIndexer = new StringIndexer()
    stringIndexer.setInputCol(column)
      .setOutputCol(s"${column}Indexed")
      .setHandleInvalid("skip")
    //and then onehotencoder
    val oneHotEncoderEstimator = new OneHotEncoderEstimator()
    oneHotEncoderEstimator.setInputCols(Array(s"${column}Indexed")).setOutputCols(Array(s"${column}OneHot"))
    Array(stringIndexer, oneHotEncoderEstimator)
  }
}

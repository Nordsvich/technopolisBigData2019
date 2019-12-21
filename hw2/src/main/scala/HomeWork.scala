import org.apache.spark.ml.{Estimator, Pipeline, Transformer}
import org.apache.spark.ml.classification.{LogisticRegression, ProbabilisticClassifier, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, Imputer, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql
import org.apache.spark.sql.{DataFrame, SparkSession}

object HomeWork {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .master("local")
      .appName("HomeWork")
      .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")

    val test = readData(spark, "test.csv")
    val train = readData(spark, "train.csv")

    train.describe().show()
    test.describe().show()

    val maxEmbarked = train
      .select("Embarked")
      .summary("max")
      .select("Embarked")
      .head()
      .getString(0)

    val embarkedTransformer = new EmbarkedTransformer(maxEmbarked);

    val sexIndexer = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("SexIndx")

    val embarkedIndexer = new StringIndexer()
      .setInputCol("Embarked")
      .setOutputCol("EmbarkedIndx")

    val imputer = new Imputer()
      .setInputCols(Array("Age"))
      .setOutputCols(Array("AgeImputed"))

    val assembler = new VectorAssembler()
      .setInputCols(Array[String]("SexIndx", "Pclass", "AgeImputed", "SibSp", "Parch", "EmbarkedIndx"))
      .setOutputCol("vectorizedFeatures")

    val selector = new ChiSqSelector()
      .setFpr(0.1)
      .setFeaturesCol(assembler.getOutputCol)
      .setLabelCol("Survived")
      .setOutputCol("selectedFeatures")

    val scaler = new StandardScaler()
      .setInputCol(selector.getOutputCol)
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("Survived")

    val rf = new RandomForestClassifier()
      .setFeaturesCol("features")
      .setLabelCol("Survived")

    val paramGridLr = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.8, 0.4, 0.1))
      .addGrid(lr.maxIter, Array(10, 15, 20))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val paramGridRf = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(5, 10, 15))
      .addGrid(rf.numTrees, Array(3, 6, 9))
      .build()

    val pipelineLr = new Pipeline()
      .setStages(Array(embarkedTransformer, sexIndexer, embarkedIndexer, imputer, assembler, selector, scaler, lr))

    val pipelineRf = new Pipeline()
      .setStages(Array(embarkedTransformer, sexIndexer, embarkedIndexer, imputer, assembler, selector, scaler, rf))

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("Survived")

    val cvLr = new CrossValidator()
      .setEstimator(pipelineLr)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGridLr)
      .setNumFolds(10)

    val cvRf = new CrossValidator()
      .setEstimator(pipelineRf)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGridRf)
      .setNumFolds(10)


    val Array(trainPart, validatePart) = train.randomSplit(Array(0.90, 0.10))

    val cvModelLr = cvLr.fit(trainPart)

    val cvModelRf = cvRf.fit(trainPart)

    val predictedLr = cvModelLr.transform(validatePart)

    val predictedRf = cvModelRf.transform(validatePart)

    val lrAUC = evaluator.evaluate(predictedLr)
    val rfAUC = evaluator.evaluate(predictedRf)

    println("LogisticRegression AUC - " + lrAUC)
    println("RandomForest AUC - " + rfAUC)

    if (lrAUC > rfAUC) {
      val predict = cvModelLr.transform(test)
      savePredict(predict)
    } else {
      val predict = cvModelRf.transform(test)
      savePredict(predict)
    }
  }

  def savePredict(predict: DataFrame) {
    predict.select("features", "rawPrediction", "probability", "prediction").show(false)

    predict.select("PassengerId", "prediction")
      .write.format("csv")
      .option("header", "true")
      .save("predict")
  }

  def readData(spark: SparkSession, path: String): sql.DataFrame = {
    spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .option("sep", ",")
      .load(path)
  }
}

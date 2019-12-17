import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, Imputer, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.IntegerType


object TitanicClass {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("elenapranova")
      .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")

    val dataTrain = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("train.csv")
    val dataTest = spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load("test.csv")

    val Array(trainOfDataTrain, testOfDataTrain) = dataTrain.randomSplit(Array(0.7, 0.3), seed = 5)

    /*dataTrain.show(5, false)
    dataTrain.describe().show(true)
    dataTest.show(5, false)
    dataTest.describe().show(true)*/

    val imputerFare = new Imputer()
      .setInputCols(Array("Fare"))
      .setOutputCols(Array("fareConvert"))

    //convert sex and age
    val indexerSex = new StringIndexer()
      .setInputCol("Sex")
      .setOutputCol("sexConvert")

    val imputerAge = new Imputer()
      .setInputCols(Array("Age"))
      .setOutputCols(Array("ageConvert"))
    //imputer.fit(dataTrain).transform(dataTrain).show(5, false)

    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "sexConvert", "ageConvert", "SibSp", "Parch", "fareConvert"))
      .setOutputCol("assebledFeatures")

    val selector = new ChiSqSelector()
      .setFpr(0.05)
      .setFeaturesCol(assembler.getOutputCol)
      .setLabelCol("Survived")
      .setOutputCol("selectedFeatures")

    val scaler = new StandardScaler()
      .setInputCol(selector.getOutputCol)
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    val rf = new RandomForestClassifier()
      .setLabelCol("Survived")
      .setFeaturesCol(scaler.getOutputCol)

    val paramGridRF = new ParamGridBuilder()
      .addGrid(rf.maxDepth, Array(5, 10, 15))
      .addGrid(rf.numTrees, Array(5, 10, 15))
      .build()

    val pipelineRF = new Pipeline()
      .setStages(Array(imputerFare, indexerSex, imputerAge, assembler, selector, scaler, rf))

    val cvRF = new TrainValidationSplit()
      .setEstimator(pipelineRF)
      .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("Survived"))
      .setEstimatorParamMaps(paramGridRF)
      //      .setNumFolds(10)
      .setTrainRatio(0.8)
      .setParallelism(4)

    val lr = new LogisticRegression()
      .setLabelCol("Survived")
      .setFeaturesCol(scaler.getOutputCol)

    val paramGridLR = new ParamGridBuilder()
      .addGrid(lr.maxIter, Array(10, 20))
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val pipelineLR = new Pipeline().setStages(Array(imputerFare, indexerSex, imputerAge, assembler, selector, scaler, lr))

    val cvLR = new TrainValidationSplit()
      .setEstimator(pipelineLR)
      .setEvaluator(new BinaryClassificationEvaluator().setLabelCol("Survived"))
      .setEstimatorParamMaps(paramGridLR)
      //      .setNumFolds(10)
      .setTrainRatio(0.8)
      .setParallelism(4)

    val rfModel = cvRF.fit(trainOfDataTrain)

    val lrModel = cvLR.fit(trainOfDataTrain)

    val rfPredict = rfModel.transform(testOfDataTrain)

    val lrPredict = lrModel.transform(testOfDataTrain)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("Survived")

    import spark.implicits._

    if (evaluator.evaluate(rfPredict)>evaluator.evaluate(lrPredict)){
        val result = rfModel.transform(dataTest)
      //result.show(true)
      //result.describe().show(true)
        //result.select("PassengerId", "prediction").show(417, false)

      result.select($"PassengerId", $"prediction".cast(IntegerType))
        .write
        .format("csv")
        .option("header", "true")
        .save("predict")
    } else {
        val result = lrModel.transform(dataTest)
      //result.show(true)
      //result.describe().show(true)
        //result.select("PassengerId", "prediction").show(417, false)


      result.select($"PassengerId", $"prediction".cast(IntegerType))
        .write
        .format("csv")
        .option("header", "true")
        .save("predict")
    }
  }
}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.avg
import org.apache.spark.sql.types.IntegerType

object Titanic extends App {

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("BigData2019_hw2")
    .getOrCreate()

  import spark.implicits._

  val trainData = spark
    .read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("titanic/train.csv")

  val testData = spark
    .read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("titanic/test.csv")

  val labelCol = "Survived"

  val ageDf = trainData.select("Age").union(testData.select("Age"))
  val avgAge = ageDf.select(avg("Age")).first().getDouble(0)

  val fareDf = trainData.select("Fare").union(testData.select("Fare"))
  val avgFare = fareDf.select(avg("Fare")).first().getDouble(0)

  val emptyMapper = Map[String, Any](
    "Age" -> avgAge,
    "Fare" -> avgFare,
    "Embarked" -> "S",
    "Cabin" -> "no")

  val trainDataPrepared = trainData.na.fill(emptyMapper)
  val testDataPrepared = testData.na.fill(emptyMapper)

  val stringColumns = Seq("Sex", "Cabin", "Embarked")
  val numColumns = Seq("Age", "SibSp", "Parch", "Fare", "Pclass")

  val indexers = stringColumns.map(colName =>
    new StringIndexer()
      .setInputCol(colName)
      .setOutputCol(colName + "Indexed")
      .setHandleInvalid("keep")
  )

  val encoder = new OneHotEncoderEstimator()
    .setInputCols(stringColumns.map(colName => s"${colName}Indexed").toArray)
    .setOutputCols(stringColumns.map(colName => s"${colName}Vec").toArray)

  val assembler = new VectorAssembler()
    .setInputCols((numColumns ++ stringColumns.map(_ + "Vec")).toArray)
    .setOutputCol("vectorizedFeatures")

  val selector = new ChiSqSelector()
    .setFeaturesCol("vectorizedFeatures")
    .setLabelCol(labelCol)
    .setOutputCol("selectedFeatures")

  val scaler = new StandardScaler()
    .setInputCol("selectedFeatures")
    .setOutputCol("scaledFeatures")
    .setWithStd(true)
    .setWithMean(true)

  val randomForestClassifier = new RandomForestClassifier()
    .setImpurity("gini")
    .setMaxDepth(3)
    .setNumTrees(20)
    .setFeatureSubsetStrategy("auto")
    .setLabelCol(labelCol)
    .setFeaturesCol("scaledFeatures")

  val pipeline = new Pipeline().setStages((
    indexers :+
      encoder :+
      assembler :+
      selector :+
      scaler :+
      randomForestClassifier).toArray)

  val model = pipeline.fit(trainDataPrepared)
  val prediction = model.transform(testDataPrepared)

  prediction
    .select($"PassengerId", $"prediction".cast(IntegerType).alias(labelCol))
    .coalesce(1)
    .write
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .csv("submission.csv")
}

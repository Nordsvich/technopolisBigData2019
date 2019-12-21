import java.util.UUID

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

class EmbarkedTransformer(val value: String) extends Transformer {
  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.na.fill(value, Seq("Embarked"))
  }

  override def copy(extra: ParamMap): Transformer = this

  override def transformSchema(schema: StructType): StructType = schema

  override val uid: String = UUID.randomUUID().toString
}

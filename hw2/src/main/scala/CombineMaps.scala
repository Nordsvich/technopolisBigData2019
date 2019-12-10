import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row

class CombineMaps[K, V](keyType: DataType, valueType: DataType, merge: (V, V) => V) extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = new StructType().add("map", dataType)
  override def bufferSchema: StructType = inputSchema
  override def dataType: DataType = MapType(keyType, valueType)
  override def deterministic: Boolean = true

  override def initialize(buffer: MutableAggregationBuffer): Unit = buffer.update(0, Map[K, V]())

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    val map1 = buffer.getAs[Map[K, V]](0)
    val map2 = input.getAs[Map[K, V]](0)
    val result = map1 ++ map2.map { case (k,v) => k -> map1.get(k).map(merge(v, _)).getOrElse(v) }
    buffer.update(0, result)
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = update(buffer1, buffer2)

  override def evaluate(buffer: Row): Any = buffer.getAs[Map[K, V]](0)
}
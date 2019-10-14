name := "FirstMLTask"

version := "0.1"

scalaVersion := "2.2.1"

val sparkVersion = "2.4.4"

val mClass = "com.github.senyast4745.firstML.MainClass"

mainClass in run := Some(mClass)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-streaming" % sparkVersion,
  "org.apache.spark" %% "spark-streaming-twitter" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion
)



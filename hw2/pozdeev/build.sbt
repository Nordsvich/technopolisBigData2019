scalaVersion := "2.12.10"
name := "BigData2019_hw2"
organization := "ru.ok.technopolis"
version := "1.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.4.4",
  "org.apache.spark" %% "spark-sql" % "2.4.4",
  "org.apache.spark" %% "spark-mllib" % "2.4.4"
)
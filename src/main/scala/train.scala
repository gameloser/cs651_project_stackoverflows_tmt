//import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.SparseVector
import org.rogach.scallop._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
//import org.apache.spark.mllib.feature.Stemmer
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.mllib.util.MLUtils
import org.apache.log4j._
import org.apache.hadoop.fs._

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.ml.classification.LogisticRegression



object NormalizeData{

  val log: Logger = {
    Logger.getLogger(getClass.getName)
  }

  def main(argv: Array[String]): Unit = {
    val args = new NormalizeConf(argv)

    log.info("Input: " + args.input())
    log.info("Output: " + args.output())

    val conf = new SparkConf().setAppName("Compute Pairs PMI")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    // get nlp-session if not created
    val spark = SparkSession.builder.enableHiveSupport().getOrCreate()

    val input_path = args.input()
    val outputPath = args.output()

    // // read csv file
    // val ques_df = spark.read.format("csv")
    //   .option("inferSchema", "true")
    //   .option("header", "true")
    //   .option("mode", "DROPMALFORMED")
    //   .load(input_path)
    //   .toDF()

    // Load training data
    val training = spark.read.format("libsvm").load(input_path)

    val lr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    

    // save to lib svm format
    MLUtils.saveAsLibSVMFile(labeledPoints, outputPath)

  }
}
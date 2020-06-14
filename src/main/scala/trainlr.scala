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
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.{RegressionEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object trainlr{

  val log: Logger = {
    Logger.getLogger(getClass.getName)
  }

  def main(argv: Array[String]): Unit = {
    val args = new NormalizeConf(argv)

    log.info("Input: " + args.input())
    log.info("Output: " + args.output())

    val conf = new SparkConf().setAppName("Train lr models")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    // get nlp-session if not created
    val spark = SparkSession.builder.enableHiveSupport().getOrCreate()

    val input_path = args.input()
    val outputPath = args.output()

    // Load training data
    val data = spark.read.format("libsvm").load(input_path)
    // training data set - 70%
    // test data set - 30%
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 2020)
    val training = splits(0).cache()
    val test = splits(1)

    val lr = new LogisticRegression()
    .setMaxIter(1000)
    .setRegParam(0.01)
    .setElasticNetParam(0.0)

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    val paramGrid = new ParamGridBuilder()
    .addGrid(lr.maxIter, Array(100, 500, 1000))
    .addGrid(lr.regParam, Array(0.1, 0.01))
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 0.8))
    .build()
    
    // evaluate model with area under ROC
    val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")
    
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    val cv = new CrossValidator()
    .setEstimator(lr)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(5)  // Use 3+ in practice
    .setParallelism(5)  // Evaluate up to x parameter settings in parallel
    
    
    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(training)

    // Fit the model
//    val lrModel = lr.fit(training)
    
    // Make predictions on test documents. cvModel uses the best model found (lrModel).
    val lrPrediction = cvModel.transform(test)
    
//    // evaluate rmse
//    val regEval = new RegressionEvaluator()
//    .setPredictionCol("prediction")
//    .setMetricName("rmse")
    

    // measure the accuracy
    val accuracy = evaluator.evaluate(lrPrediction)
    
    println(s"areaUnderROC: ${accuracy}")
    
//    println("the mean absolute error is  "+ regEval.evaluate(lrPrediction))
//    println("the root mean squared error is " + regEval.evaluate(lrPrediction))

  }
}

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
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.RegressionEvaluator

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier, GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}


object trainrgbdt{

  val log: Logger = {
    Logger.getLogger(getClass.getName)
  }

  def main(argv: Array[String]): Unit = {
    val args = new NormalizeConf(argv)

    log.info("Input: " + args.input())
    log.info("Output: " + args.output())

    val conf = new SparkConf().setAppName("Train models")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    // get nlp-session if not created
    val spark = SparkSession.builder.enableHiveSupport().getOrCreate()
    
    val input_path = args.input()
    val outputPath = args.output()

    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load(input_path)
    
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)
    
    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
    
    // Train a GBT model.
    val gbt = new GBTClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)
      .setFeatureSubsetStrategy("auto")
    
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    
    // Chain indexers and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
    
    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)
    
    // Make predictions.
    val predictions = model.transform(testData)
    
    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)
    
    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1.0 - accuracy}")

    //val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
    //println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
  }
}

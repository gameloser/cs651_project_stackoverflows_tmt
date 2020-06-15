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

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}


object trainrf{

  val log: Logger = {
    Logger.getLogger(getClass.getName)
  }

  def main(argv: Array[String]): Unit = {
    val args = new NormalizeConf(argv)

    log.info("Input: " + args.input())
    log.info("Output: " + args.output())

    val conf = new SparkConf()
    .setAppName("Train models")
    .set("spark.driver.maxResultSize", "4g")
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
      .setMaxCategories(5)
      .fit(data)
    
    // Split the data into training and test sets (10% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 2020)
    
    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setSeed(2020)
    
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    
    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    
    val paramGrid = new ParamGridBuilder()
    .addGrid(rf.numTrees, Array(10))
//    .addGrid(rf.maxDepth, Array(3, 5, 8))
    .build()
    
    // Train model. This also runs the indexers.
//    val rfmodel = pipeline.fit(trainingData)
    
    // Make predictions.
//    val rfPrediction = rfmodel.transform(testData)
    
  
    // Select example rows to display.
//    rfPrediction.select("predictedLabel", "label", "features").show(5)
    
//    // Select (prediction, true label) and compute test error.
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("indexedLabel")
//      .setPredictionCol("prediction")
//      .setMetricName("accuracy")
    
    // evaluate model with area under ROC
    val evaluator = new BinaryClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setRawPredictionCol("prediction")
    .setMetricName("areaUnderROC")
    
    val trainValidationSplit = new TrainValidationSplit()
    .setEstimator(pipeline)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    // 80% of the data will be used for training and the remaining 20% for validation.
    .setTrainRatio(0.8)
    // Evaluate up to 2 parameter settings in parallel
    .setParallelism(3)
    
    // Run train validation split, and choose the best set of parameters.
    val rfmodel = trainValidationSplit.fit(trainingData)
    
    // Make predictions on test documents. rfModel uses the best model found.
    val lrPrediction = rfmodel.transform(testData)
    
    // measure the accuracy
    val accuracy = evaluator.evaluate(lrPrediction)
    println(s"areaUnderROC: ${accuracy}")
//    println(s"Test Error = ${(1.0 - accuracy)}")
    
  }
}

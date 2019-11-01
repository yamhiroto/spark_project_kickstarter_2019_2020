package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator



object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")

    /** BUILDING PIPELINE **/

    // Load dataset

    val df = spark.read.load("./spark_project_kickstarter_2019_2020")

    //STAGE 1 - RETRIEVE WORDS (TOKEN) FROM TEXT

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")


    //STAGE 2 - REMOVE STOP WORDS

    val tokenFilter = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("tokens_filtered")


    //STAGE 3 - VECTORIZE TOKEN - TF

    val tokenVectorizer = new CountVectorizer()
      .setInputCol("tokens_filtered")
      .setOutputCol("tokens_vectorized")


    //STAGE 4 - COMPUTE IDF

    val idf = new IDF()
      .setInputCol("tokens_vectorized")
      .setOutputCol("tfidf")


    //STAGE 5 - TRANSFORM country2 FROM CATEGORY VARIABLE TO NUMERICAL VARIABLE

    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")


    //STAGE 6 - TRANSFORM currency2 FROM CATEGORY VARIABLE TO NUMERICAL VARIABLE

    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")


    //STAGE 7 - TRANSFORM country_indexed FROM NUMERICAL VARIABLE WITH A ONE-HOT ENCODING

    val country_encoder = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_onehot")


    //STAGE 8 - TRANSFORM currency_indexed FROM NUMERICAL VARIABLE WITH A ONE-HOT ENCODING

    val currency_encoder = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_onehot")


    //STAGE 9 - GATHER ALL THE FEATURES IN ONE COLUMN features

    val features_assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")


    //STAGE 10 - INSTANTIATE CLASSIFICATION MODEL

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    /** BUILD PIPELINE */

    //Create Pipeline

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, tokenFilter, tokenVectorizer, idf, country_indexer, currency_indexer,
        country_encoder, currency_encoder, features_assembler, lr))


    //Split data in training and test sets

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), 0)


    //Model training

    val trained_model: PipelineModel = pipeline.fit(training)


    //Test Model

    val dfWithSimplePredictions : DataFrame = trained_model.transform(test)


    //Display f1-score for dfWithSimplePredictions

    val f1Evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    println("f1 score : " + f1Evaluator.evaluate(dfWithSimplePredictions))

    println("Result for dfWithSimplePredictions")
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()


    /**  FEATURES TUNING  **/

    // GridSearch
    
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(tokenVectorizer.minDF, Array(55.0,75.0,95.0))
      .build()


    // Train-validation split using 70% as training and F1-score as evaluator

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)


    // Model Training

    val model = trainValidationSplit.fit(training)


    // Test Model

    val dfWithPredictions = model.transform(test)
      .select("features", "final_status", "predictions")


    //Display f1-score for dfWithSimplePredictions

    val f1_score = evaluator.evaluate(dfWithPredictions)

    println("f1-score on test set: " + f1_score)


    // Display predictions

    println("Result :")
    dfWithPredictions.groupBy("final_status", "predictions").count.show()


  }
}

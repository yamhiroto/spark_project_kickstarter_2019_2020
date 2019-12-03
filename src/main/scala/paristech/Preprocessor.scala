package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}


object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._
    /** *****************************************************************************
      *
      * TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    println("\n")
    println("Hello World ! from Preprocessor")
    println("\n")


    /** LOADING DATA **/

    // load train_clean.csv into a DataFrame

    val df: DataFrame = spark
      .read
      .option("header", true) // Use first line of files as header
      .option("inferSchema", "true") // to infer the data type of each column (Int, String, etc.)
      .csv("src/main/resources/train/train_clean.csv")

    // display number of rows and colums within the DataFrame

    println(s"Total number of rows: ${df.count}")
    println(s"Number of columns ${df.columns.length}")

    // display first rows of the DataFrame
    df.show()

    // display DataFrame's schema (column's name along with its type)
    df.printSchema()

    // Cast type as "Integer" to relevant columns
    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline", $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()


    /** 2 - CLEANING DATA **/

    // display a few information on columns whose type is Integer
    dfCasted
      .select("goal", "backers_count", "final_status")
      .describe()
      .show

    // Display DataFrame information to see how and what to clean
   /**
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)
     **/

    //  drop column 'disable_communication, since it doesn't provide much information (only 322 rows are True, which is negligible)
    val df2: DataFrame = dfCasted.drop("disable_communication")

    // drop columns to prevent any data_leakage
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    // display country and currency columns, we can see that when 'country' === False, 'country' value is within 'currency' column
    df.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)

    //UserDefinedFonction UDF  cleanCountry to clean column 'country'
    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else
        country
    }

    //UserDefinedFonction UDF  cleanCurrency to clean column 'currency'
    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    //replace columns 'country' and 'currency' by their cleaned version 'country2' and 'currency2'
    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    /** OTHER SOLUTIONS :
      * import sql.functions.when
      * dfNoFutur
      * .withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
      * .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
      * .drop("country", "currency")
      */

    //display number of element of each class (column 'final_status)
    dfCountry.groupBy("final_status").count.orderBy($"count".desc).show(30)

    // filter DataFrame to only keep rows whose final_status is either 0 (Fail) or 1 (Success)
    val dfFilteredStatus: DataFrame = dfCountry.filter($"final_status".isin(0, 1))


    /** - FEATURE ENGINEERING **/

    //convert 'date' columns from timestamps Unix to a String

    val dfTime: DataFrame = dfFilteredStatus
      .withColumn("deadline2", from_unixtime($"deadline"))
      .withColumn("created_at2", from_unixtime($"created_at"))
      .withColumn("launched_at2", from_unixtime($"launched_at"))

    //dfTime.show(10)

    //add column days_campaign
    val dfDays_campaign: DataFrame = dfTime
      .withColumn("days_campaign", datediff($"deadLine2" , $"launched_at2"))

      //add column hours_campaign
    val dfHours_campaign: DataFrame = dfDays_campaign
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at") / 3600.0, 3))
      .drop("created_at", "deadline", "launched_at")

    // convert values (string) within 'name', 'desc' and 'keywords' columns to lower case
    val dfLower: DataFrame = dfHours_campaign
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))

    //dfLower.show(10)

    //concatenate string columns
    val dfConcat = dfLower
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))
      .drop("name", "desc", "keywords")

    dfConcat.show(220)

    /** - NULL VALUES **/

    //remove null values
    val dfFinal: DataFrame = dfConcat
      .na.fill(-1, Array("days_campaign","hours_prepa","goal"))
      .na.fill("unknown", Array("country2","currency2"))


    dfFinal.show(50)


    /** 5 - SAVE DATAFRAME **/

    dfFinal.write.mode(SaveMode.Overwrite).parquet("./src/main/resources/preprocessed")


  }
}

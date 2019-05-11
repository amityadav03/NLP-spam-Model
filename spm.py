#Use SMS data to build a Spam Detection model.

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('nlp').getOrCreate()
data = spark.read.csv("spam.csv",inferSchema=True)
data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')
data.show()

# Check for any missing values
from pyspark.sql.functions import isnan, isnull, when, count, col

data.select([count(when(isnan(c)| isnull(c), c)).alias(c) for c in data.columns]).show()

# Create a length column containing the text length
from pyspark.sql.functions import length

data = data.withColumn('length',length(data['text']))
data.show()

# Calculate the mean ham and spam length
data.groupby('class').mean().show()
from pyspark.ml.feature import (Tokenizer,StopWordsRemover,CountVectorizer,
                                IDF,StringIndexer)

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
class_to_num = StringIndexer(inputCol='class',outputCol='label')

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector

assembler = VectorAssembler(inputCols=['tf_idf','length'],
                            outputCol='features')
# Use Pipeline since there are a number of steps for data preparation

from pyspark.ml import Pipeline

data_prep_pipe = Pipeline(stages=[class_to_num,
                                  tokenizer,
                                  stopremove,
                                  count_vec,
                                  idf,
                                  assembler])
final_data = data_prep_pipe.fit(data).transform(data)
final_data = final_data.select(['label','features'])
final_data.show()

# Split the data 7:3

train,test = final_data.randomSplit([0.7,0.3])

# We will use the classification model Naive Bayes

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes()

spam_detector = nb.fit(train)

data.printSchema()

final_data.printSchema()

results = spam_detector.transform(test)

results.show()

# Evaluate the results using a MulticlassClassificationEvaluator

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(results)
print("Accuracy of model at detecting spam was: {}".format(acc))
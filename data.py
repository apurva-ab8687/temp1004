import sys
import itertools
# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.types import *

from operator import itemgetter
from itertools import groupby
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
memory = '3g' #giving spark all available memory minus 10% overhead
spark = (SparkSession.builder
         .appName('train_als')
         .master('yarn')
         .config('spark.executor.memory', memory)
         .config('spark.driver.memory', memory)
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# Read file
file_name = 'hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv'
df = spark.read.csv(file_name, header=True)
      
df.createOrReplaceTempView('data')
data_five_percent = spark.sql('select * from data where int(user_id)%20 = 0')

# Sample 5% of the total data and convert file to parquet format  
#df = df.sample(False, 0.05, 20)                                    
data_five_percent.write.parquet('datadir_five.parquet')
#df_output = df.coalesce(1)                                                  
#df_output.show()  

#Select records having user_id has given more than 10 reviews
data_five_percent.createOrReplaceTempView('parquetFile')
clean_data = spark.sql("SELECT * FROM parquetFile WHERE user_id IN ( SELECT user_id FROM parquetFile GROUP BY user_id HAVING COUNT(DISTINCT book_id)>=10)")
#clean_data.count()
#df_output.count()

#Selecting distinct user_id
clean_data.createOrReplaceTempView('cleanfile')
user_list = spark.sql("SELECT DISTINCT user_id FROM cleanfile")
#user_list.count()

#Split data into train, test validation based on user_id
from sklearn.model_selection import train_test_split
train_users, val_users, test_users = user_list.randomSplit([0.6, 0.2, 0.2])
train_users, val_users, test_users = train_users.collect(), val_users.collect(), test_users.collect()
#len(train_users), len(val_users), len(test_users)                           
train_users = [int(x[0]) for x in train_users]
val_users = [int(x[0]) for x in val_users]
test_users = [int(x[0]) for x in test_users]

# Select data from val and test for merging into train set based on interactions
val_data = spark.sql("SELECT * FROM cleanfile WHERE user_id IN {0}".format(tuple(val_users)))
test_data = spark.sql("SELECT * FROM cleanfile WHERE user_id IN {0}".format(tuple(test_users)))
train_data = spark.sql("SELECT * FROM cleanfile WHERE user_id IN {0}".format(tuple(train_users)))


# split 50 % of the data from val and test data for merge into train set

from pyspark.sql.window import Window
import pyspark.sql.functions as F

window = Window.partitionBy('user_id').orderBy('book_id')
val_data = (val_data.select("user_id", "book_id", "is_read", "rating", "is_reviewed",
                  F.row_number()
                  .over(window)
                  .alias("row_number")))

val_data.createOrReplaceTempView('valfile')
val_train_data = spark.sql("SELECT user_id, book_id, is_read, rating, is_reviewed FROM valfile WHERE row_number%2==0")
val_data = spark.sql("SELECT user_id, book_id, is_read, rating, is_reviewed FROM valfile WHERE row_number%2!=0")

window = Window.partitionBy('user_id').orderBy('book_id')
test_data = (test_data.select("user_id", "book_id", "is_read", "rating", "is_reviewed",
                  F.row_number()
                  .over(window)
                  .alias("row_number")))

test_data.createOrReplaceTempView('testfile')
test_train_data = spark.sql("SELECT user_id, book_id, is_read, rating, is_reviewed FROM testfile WHERE row_number%2==0")
test_data = spark.sql("SELECT user_id, book_id, is_read, rating, is_reviewed FROM testfile WHERE row_number%2!=0")

# Merge 50 % of the data from val and test data for merge into train set
train_1 = train_data.union(val_train_data)
train_2 = train_1.union(test_train_data)

# Change type string to int
from pyspark.sql.types import *
col_list = ['user_id', 'book_id', 'is_read', 'rating', 'is_reviewed']
for col in col_list:
    train_1 = train_1.withColumn(col, train_1[col].cast(IntegerType()))
    train_2 = train_2.withColumn(col, train_2[col].cast(IntegerType()))
    val_data = val_data.withColumn(col, val_data[col].cast(IntegerType()))
    test_data = test_data.withColumn(col, test_data[col].cast(IntegerType()))



train_1.write.parquet('train_1_five.parquet')
#train_1 = train_1.coalesce(1)                                                  
train_2.write.parquet('train_2_five.parquet')
#train_2 = train_2.coalesce(1)                                                  
val_data.write.parquet('val_data_five.parquet')
#val_data = val_data.coalesce(1)                                                  
test_data.write.parquet('test_data_five.parquet')
#test_data = test_data.coalesce(1)


#path_ = 'hdfs://dumbo/user/ab8687/train_1.parquet'
train_1= spark.read.parquet('hdfs://dumbo/user/ab8687/train_1_five.parquet')
#train_1.show(5)
#train_1.printSchema()
#path = 'hdfs://dumbo/user/ab8687/train_2.parquet'
train_2= spark.read.parquet('hdfs://dumbo/user/ab8687/train_2_five.parquet')
#train_2.show(5)
#train_2.printSchema()

#path = 'hdfs://dumbo/user/ab8687/val_data.parquet'
val_data= spark.read.parquet('hdfs://dumbo/user/ab8687/val_data_five.parquet')
#val_data.show(5)
#val_data.printSchema()

#path = 'hdfs://dumbo/user/ab8687/test_data.parquet'
test_data= spark.read.parquet('hdfs://dumbo/user/ab8687/test_data_five.parquet')
#test_data.show(5)
#test_data.printSchema()


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
from pyspark.sql.functions import expr
from operator import itemgetter
from itertools import groupby
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import expr
memory = '3g' #giving spark all available memory minus 10% overhead
spark = (SparkSession.builder
         .appName('train_als')
         .master('yarn')
         .config('spark.executor.memory', memory)
         .config('spark.driver.memory', memory)
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
#path_ = 'hdfs://dumbo/user/ab8687/train_1.parquet'
train_1= spark.read.parquet('hdfs://dumbo/user/ab8687/train_1_five.parquet')
#train_1.show(5)
train_1.printSchema()


#path = 'hdfs://dumbo/user/ab8687/val_data.parquet'
val_data= spark.read.parquet('hdfs://dumbo/user/ab8687/val_data_five.parquet')
#val_data.show(5)
val_data.printSchema()

user_id_val = val_data.select('user_id').distinct()
true_label = val_data.select('user_id', 'book_id').groupBy('user_id').agg(expr('collect_list(book_id) as true_item'))
    
ranks       = [10,20,30,40]
lambdas     = [0.5,0.7,1]
numIters    = [1,3,5]
temp = []
result = []
bestModel   = None
bestRmse = float("inf")
bestMAP=0.0
bestRank    = 0
bestLambda  = -1.0
bestNumIter = -1
for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
    als=ALS(maxIter=numIter,regParam=lmbda,userCol="user_id",itemCol="book_id",ratingCol="rating",
            coldStartStrategy="drop",implicitPrefs=False, rank=rank)
    model = als.fit(train_1)
    predictions = model.transform(val_data)
    top500=model.recommendForAllUsers(500)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    temp.append(rank)
    temp.append(lmbda)
    temp.append(numIter)
    temp.append(rmse)
    result.append(temp)
    temp = []
    # Make top 500 recommendations for users in validation test
    res = model.recommendForUserSubset(user_id_val,500)
    pred_label = res.select('user_id','recommendations.book_id')
    pred_true_rdd = pred_label.join(F.broadcast(true_label), 'user_id', 'inner').rdd.map(lambda row: (row[1], row[2]))
    print('Start Evaluating for ',"Rank",rank, "Lambda",lmbda, "Iteration", numIter)
    metrics = RankingMetrics(pred_true_rdd)
    print("markerappapappa")
    mpa = metrics.precisionAt(500)
    print ("Rank",rank, "Lambda",lmbda, "Iteration", numIter, "RMSE",rmse, 'map score: ', mpa)
print ("Bestrank",bestRank, "BestLambda", bestLambda, "BestIter",bestNumIter, "Bestrmse",bestRmse,"BestMAP",bestMAP) 
print ("ALS on train:\t\t%.2f" % bestRmse)
print ("ALS on train MAP:\t\t%.2f" % bestMAP)



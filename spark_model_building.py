# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 13:47:56 2020

@author: bkorzen
"""

from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('ObesityRegression').getOrCreate()
data = spark.read.csv('obesity_dummy.csv', inferSchema=True, header=True)

data.columns

feature_assembler = VectorAssembler(inputCols=['age',
 'veges_freq_Always',
 'veges_freq_Never',
 'veges_freq_Sometimes',
 'main_meals_num_More than 3',
 'main_meals_num_One/Two',
 'main_meals_num_Three',
 'daily_water_consumption_1-2L',
 'daily_water_consumption_<1L',
 'daily_water_consumption_>2L',
 'physical_activity_freq_0 days',
 'physical_activity_freq_1-2 days',
 'physical_activity_freq_2-4 days',
 'physical_activity_freq_4-5 days',
 'tech_devices_usage_0-2h',
 'tech_devices_usage_3-5h',
 'tech_devices_usage_>5h',
 'high_kcal_food_no',
 'high_kcal_food_yes',
 'transport_used_Automobile',
 'transport_used_Bike',
 'transport_used_Motorbike',
 'transport_used_Public_Transportation',
 'transport_used_Walking',
 'snacks_consuming_Always',
 'snacks_consuming_Frequently',
 'snacks_consuming_Sometimes',
 'snacks_consuming_no',
 'smoking_no',
 'smoking_yes',
 'kcal_monitoring_no',
 'kcal_monitoring_yes',
 'alcohol_consumption_Always',
 'alcohol_consumption_Frequently',
 'alcohol_consumption_Sometimes',
 'alcohol_consumption_no',
 'gender_Female',
 'gender_Male'], outputCol='features')

output = feature_assembler.transform(data)

output.select("features").show()

final_data = output.select('features', 'bmi')
# final_data.show()

train_data, test_data = final_data.randomSplit([0.8,0.2])

regressor=LinearRegression(featuresCol='features', labelCol='bmi')
regressor=regressor.fit(train_data)

regressor.coefficients
regressor.intercept
pred_results=regressor.evaluate(test_data)
pred_results.predictions.show()

evaluator = RegressionEvaluator(
    labelCol="bmi", predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(pred_results.predictions)
print("Linear Regression MAE: " + str(mae))
print("#######################")
rf = RandomForestRegressor(featuresCol="features", labelCol='bmi')
# pipeline = Pipeline(stages=[featureAssembler, rf])
model = rf.fit(train_data)
predictions = model.transform(test_data)
predictions.show()

evaluator = RegressionEvaluator(
    labelCol="bmi", predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictions)
print("Random Forest Regression MAE: " + str(mae))



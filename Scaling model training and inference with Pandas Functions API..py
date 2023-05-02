# Databricks notebook source
# MAGIC %md
# MAGIC # Training with Pandas Function API
# MAGIC
# MAGIC This notebook demonstrates how to use Pandas Function API to manage and scale machine learning models for individual products. 
# MAGIC
# MAGIC  - Use [`.groupBy().applyInPandas()`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.GroupedData.applyInPandas.html?highlight=applyinpandas#pyspark.sql.GroupedData.applyInPandas) to build many models in parallel for each product.

# COMMAND ----------

# MAGIC %md
# MAGIC The dummy data we will use have the following fields:
# MAGIC - `record_id`: 100k unique records
# MAGIC - `product_id`: 10 different products
# MAGIC - `numeric_feature_1`: a numeric feature 
# MAGIC - `numeric_feature_2`: another numeric feature
# MAGIC - `label`: numeric column we are trying to predict. In practical world this can be total sales, conversion etc.

# COMMAND ----------

from pyspark.sql.functions import rand
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct

# Define the number of products and records
numProducts = 10
numRecords = 100000

# Create a DataFrame with product IDs
products = list(range(1, numProducts + 1))
productDF = spark.createDataFrame([(x,) for x in products], ["product_id"])

# Create a DataFrame with record IDs
records = list(range(1, numRecords + 1))
recordDF = spark.createDataFrame([(x,) for x in records], ["record_id"])

# Generate random values for the two numeric features
numeric_feature1 = rand(seed=42)
numeric_feature2 = rand(seed=1234)

# Combine the DataFrames and add a label column with random values
df = recordDF.crossJoin(productDF)
df = df.withColumn("numeric_feature1", numeric_feature1)
df = df.withColumn("numeric_feature2", numeric_feature2)
df = df.withColumn("label", rand(seed=5678))

# Show the resulting DataFrame
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Define the return schema

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType

trained_models_info_schema = StructType([
  StructField("product_id", IntegerType()), 
  StructField("training_sample_size", IntegerType()),    
  StructField("model_path", StringType()), 
  StructField("rmse", FloatType())          
])

# COMMAND ----------

import mlflow

#lets create a mlflow experiment to track our model training
#by default each python model has its own experiment however we can define our own

# Define the name of the experiment to create
# this has to be absolute path
experiment_name = "/Shared/product_forecasting_experiment"

# Check if the experiment already exists
existing_experiment = mlflow.get_experiment_by_name(experiment_name)

# If the experiment exists, delete it
if existing_experiment:
    mlflow.delete_experiment(existing_experiment.experiment_id)

# Create a new experiment with the given name
experiment_id = mlflow.create_experiment(experiment_name)

print(experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC Create a Pandas function that takes all the data for a specified device, trains a model, saves it as a nested run, and returns a Spark object with the defined schema.

# COMMAND ----------

import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
  """
  Trains GradientBoostingRegressor model on a group of data 
  """
  #collect information about the current dataframe that is being processed
  #get the product_id for which model is being trained
  product_id = df_pandas["product_id"].iloc[0]
  #get the number of records in the dataframe
  training_sample_size = df_pandas.shape[0]
  #get the run_id for the current MLflow run for logging later
  run_id = df_pandas["run_id"].iloc[0] 
  #get experiment_id so that we can log all the runs under that experiment
  experiment_id = df_pandas["experiment_id"].iloc[0]
  
  # Create the Gradient Boosting Regression model
  gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=42)
  
  # Define features to train on and the label
  X = df_pandas[["numeric_feature1", "numeric_feature2"]]
  y = df_pandas["label"]
  
  # Split the data into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

  # Train the model on the training data
  gbr.fit(X_train, y_train)

  # Evaluate model
  predictions = gbr.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, predictions)) 

  # we will use the top level run_id so that the current model can be logged as part of it
  with mlflow.start_run(run_id=run_id, experiment_id=experiment_id) as umbrella_run:
    print(f"Current experiment_id = {experiment_id}")
    
    # Create a nested run for the specific product
    with mlflow.start_run(run_name=str(product_id), nested=True, experiment_id=experiment_id) as run:
      mlflow.sklearn.log_model(gbr, str(product_id))
      mlflow.log_metric("rmse", rmse)
      
      artifact_uri = f"runs:/{run.info.run_id}/{product_id}"
      
      return_df = pd.DataFrame([[product_id, training_sample_size, artifact_uri, rmse]], 
        columns=["product_id", "training_sample_size", "model_path", "rmse"])

  return return_df 


# COMMAND ----------

# MAGIC %md
# MAGIC Apply the pandas function to grouped data.
# MAGIC
# MAGIC In the example we're working with, we'll be reusing the training data, which includes the product IDs, run IDs, and experiment IDs.

# COMMAND ----------

from pyspark.sql.functions import lit
#explicitly set the experiment for the moddel trainings we are going to perform
with mlflow.start_run(run_name="Model Training for all the products", experiment_id=experiment_id) as run:
  run_id = run.info.run_id
    
  model_training_info_df = (df
    .withColumn("run_id", lit(run_id)) 
    .withColumn("experiment_id", lit(experiment_id))
    .groupby("product_id")
    .applyInPandas(train_model, schema=trained_models_info_schema)
    .cache()
  )
  
display(model_training_info_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Create a pandas function to apply the model, which only requires a single read from DBFS for each product_id.

# COMMAND ----------

prediction_schema = StructType([
  StructField("record_id", IntegerType()),
  StructField("prediction", FloatType())
])

combined_df = (df
  .join(model_training_info_df, on="product_id", how="left")
)

display(combined_df)

# COMMAND ----------

def predict(df_pandas: pd.DataFrame) -> pd.DataFrame:
  """
  Applies model to data for a particular product, represented as a pandas DataFrame
  """
  model_path = df_pandas["model_path"].iloc[0]
  
  X = df_pandas[["numeric_feature1", "numeric_feature2"]]
  
  model = mlflow.sklearn.load_model(model_path)
  prediction = model.predict(X)
  
  result_df = pd.DataFrame({
    "record_id": df_pandas["record_id"],
    "prediction": prediction
  })
  return result_df

predictions_df = combined_df.groupby("product_id").applyInPandas(predict, schema=prediction_schema)
display(predictions_df)

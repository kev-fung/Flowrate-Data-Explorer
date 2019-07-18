# Databricks notebook source
# DBTITLE 1,Git Repository Management
# MAGIC %sh 
# MAGIC 
# MAGIC cwd=$(pwd)
# MAGIC does_my_git_repo_exist="$cwd/acse-9-independent-research-project-kkf18"
# MAGIC 
# MAGIC if [ -d $does_my_git_repo_exist ] 
# MAGIC then
# MAGIC   echo "Git repo exists!"
# MAGIC   echo "Pulling master branch..."
# MAGIC   cd acse-9-independent-research-project-kkf18
# MAGIC   git pull origin master
# MAGIC else
# MAGIC   git clone https://github.com/msc-acse/acse-9-independent-research-project-kkf18.git
# MAGIC   echo "Git repo does not exist!"
# MAGIC   echo "Cloning repo..."
# MAGIC fi

# COMMAND ----------

# DBTITLE 1,Repo Paths Setup for Import
import os
import sys
import importlib.util
from pathlib import Path


# Add repo path to our sys.path for importing modules from repo.
def import_mod(module_name):
  cwd = os.getcwd()
  my_git_repo_exists = Path('{}/acse-9-independent-research-project-kkf18'.format(cwd))

  spec = importlib.util.spec_from_file_location("{}.py".format(module_name), "{}/{}.py".format(my_git_repo_exists, module_name))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  
  # load module into the sys module dictionary so it can be imported in
  sys.modules[module_name] = module
  
  print("Import successful")
  
  assert module_name in sys.modules.keys()


# COMMAND ----------

# DBTITLE 1,Import Packages and Repo Modules
import numpy as np
import matplotlib as plt
import pandas as pd

# Homemade Modules
import_mod("Data_Engineering")
import Data_Engineering as DET


# COMMAND ----------

# DBTITLE 1,Workbench
# Load data into the notebook
df_01 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/newdump_01.csv')
df_02 = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/newdump_02.csv')

# Rename and cast types for each column
df_01 = df_01.select(
      df_01["Unnamed: 0"].alias("index"),
      F.to_timestamp(F.col("ts").cast("string"), "dd-MMM-yy HH:mm:ss").alias("datetime"),
      df_01["name"].alias("tag"),
      df_01["value"]
)

df_02 = df_02.select(
      df_02["Unnamed: 0"].alias("index"),
      F.to_timestamp(F.col("ts").cast("string"), "dd-MMM-yy HH:mm:ss").alias("datetime"),
      df_02["name"].alias("tag"),
      df_02["value"]
)


# Clean up data using Data Engineering Tools
DataEng = DET.GroupDataTools(df_01)
DataEng.append_data(df_02)
DataEng.is_null(explore.df)
DataEng.df = explore.null2zero("value", explore.df)
DataEng.is_null(explore.df)

# COMMAND ----------


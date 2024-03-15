#import datetime
from datetime import datetime, timedelta, date

import pyspark
import pyspark.sql.functions as f
from pyspark import SparkContext
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings('ignore')


#nohup spark-submit --driver-memory 100G --executor-cores 10 --conf spark.driver.maxResultSize=15g --conf spark.rapids.sql.enabled=true 20220325_CalculateThresholdsStefania.py &

spark = SparkSession.builder.appName("GenerateSignaturesCancan2022OriginalSplit").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
sc = spark.sparkContext






VarSourceParquetFile = "/rapids/notebooks/LICIT_COMMON_Folder/LICIT_INPUT/notebooks/tools/DetectionParquets/runPaper/"





# load source data
VarSparkDF = spark.read.parquet(VarSourceParquetFile)

listDSPFColumns = VarSparkDF.columns
if "__index_level_0__"  in listDSPFColumns:
    print("Stefania's output detected")
    VarSparkDF = VarSparkDF.drop("__index_level_0__")
    VarSparkDF = VarSparkDF.withColumn('week_of_year', f.col('week_of_year') + f.lit(1))

VarSparkDF.printSchema()



# generate week groups


lGroups = []
for i in range(0,3):
    lGroups.append([i])

dfGroups = sc.parallelize(lGroups).toDF(["WeeksGroup"])
dfGroups = dfGroups.withColumn("WeeksGroup", dfGroups.WeeksGroup.cast('integer'))
#dfGroups.show()


dfTestWeekGroups = sc.parallelize([
                     [0, 12],
                     [0, 13],
                     [0, 14],
                     [0, 15],
                     [1, 16],
                     [1, 17],
                     [1, 18],
                     [1, 19],
                     [2, 20],
                     [2, 21],
                     [2, 22],
                     [2, 23],
                     [2, 24] ]).toDF(['TestWeeksGroup', 'TestWeek'])
#dfTestWeekGroups.show()

JoinDataTestWeeks = dfTestWeekGroups.crossJoin(dfGroups)

TrainWeeks = JoinDataTestWeeks.filter(JoinDataTestWeeks.TestWeeksGroup!=JoinDataTestWeeks.WeeksGroup).withColumnRenamed("TestWeek","TrainWeek").drop("TestWeeksGroup")
TrainWeeks = TrainWeeks.withColumn("TrainWeek", TrainWeeks.TrainWeek.cast('integer'))


lGroups = []
for i in range(0,3):
    lGroups.append([i])

dfGroups = sc.parallelize(lGroups).toDF(["TrainingWeeksGroup"])
dfGroups = dfGroups.withColumn("TrainingWeeksGroup", dfGroups.TrainingWeeksGroup.cast('integer'))


TrainWeeks.show()


JoinData = VarSparkDF.crossJoin(TrainWeeks)
JoinData.printSchema()


TrainingSPDF = JoinData.filter( JoinData.week_of_year==JoinData.TrainWeek).drop('TrainWeek').repartition("WeeksGroup", "LocationId")








def level1byPL(alr:pd.Series) -> float:
    """Sets a level 1 ALR threshold (i.e. pre-alert threshold) for the input data series.
    The threshold corresponds to the 2-sigma quantile of the ALR distribution.

    Parameters
    ----------
    alr: pandas.core.frame.Series
        The series of ALR values
    
    Returns
    -------
    thresh: float
        The ALR threshold of the series
    """

    #res = alr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    # 1 alert every 22 minutes was the parameter used by 
    # once per 4 hours -> 1 per 4*60 minutes
    #return res.quantile(1/240)
    return alr.quantile(0.00416667)

def level2byPL(alr:pd.Series) -> float:
    # 1 alert every 370 minutes
    """Sets a level 2 ALR threshold (i.e. alert threshold) for the input data series.
    The threshold corresponds to the 3-sigma quantile of the ALR distribution.

    Parameters
    ----------
    alr: pandas.core.frame.Series
        The series of ALR values
    
    Returns
    -------
    thresh: float
        The ALR threshold of the series
    """

    #res = alr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    #return res.quantile(1-0.9973)
    # once per day -> 1 per 24*60 minutes
    #return res.quantile(1/1440)
    return alr.quantile(0.0006944)

def level3byPL(alr:pd.Series) -> float:
    """Sets a level 3 ALR threshold (i.e. maximal alert threshold) for the input data series.
    The threshold corresponds to the 4-sigma quantile of the ALR distribution.

    Parameters
    ----------
    alr: pandas.core.frame.Series
        The series of ALR values
    
    Returns
    -------
    thresh: float
        The ALR threshold of the series
    """
    #1 alert every 15873 minute
    #res = alr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    #return res.quantile(1-0.999937)
    return alr.quantile(0.0000992)





def level1byPL6Thresh(alr:pd.Series) -> float:
    return alr.quantile(1./240.)
def level2byPL6Thresh(alr:pd.Series) -> float:
    return alr.quantile(1./480.)
def level3byPL6Thresh(alr:pd.Series) -> float:
    return alr.quantile(1./(12.*60.))
def level4byPL6Thresh(alr:pd.Series) -> float:
    return alr.quantile(1./(24.*60.))
def level5byPL6Thresh(alr:pd.Series) -> float:
    return alr.quantile(1./(24.*60.*2.))
def level6byPL6Thresh(alr:pd.Series) -> float:
    return alr.quantile(1./(24.*60.*7.))





# we have to process the thresholds in one single percentile_approx operation, otherwise the script seems to crash....

ThreshList = [1 - (1./240.), 1 - (1./480.), 1 - (1./(12.*60.)), 1 - (1./(24.*60.)), 1 - (1./(24.*60.*2.)), 1 - (1./(24.*60.*7.)) ]


# data is ranked top to bottom
# so the highest error percentiles are the highest figures
thresholdsSpark = TrainingSPDF.groupBy('WeeksGroup', 'LocationId').agg(
    percentile_approx('var', ThreshList).alias('SixThreshLevels')
)


thresholdsSpark.printSchema()




#thresholdsSpark.to_parquet(path='exports/20220506_ThesholdsStefania_acc_6Thresh_parquet', partition_cols=['WeeksGroup'], index=False)


thresholdsSpark.repartition("WeeksGroup", "LocationId").write.partitionBy("WeeksGroup", "LocationId").option('header', 'true').parquet('exports/20220506_ThesholdsStefania_acc_6Thresh_parquet')


exit()



# stuff to separate into columns

import pandas as pd

df = spark.read.parquet('exports/20220506_ThesholdsStefania_acc_6Thresh_parquet')
for i in range(0,6):
    df = df.withColumn('level' + str(i+1) + 'byPL6Thresh', df['SixThreshLevels'].getItem(i))

df = df.drop('SixThreshLevels')

df.printSchema()

dfp = df.toPandas()

print(dfp.info(verbose=True))

dfp.to_csv('exports/20220506_ThesholdsStefania_acc_6Thresh.csv', index=False)


exit()





thresholds = thresholdsSpark.toPandas()


#MedianSpark = TrainingSPDF.groupBy('WeeksGroup', 'LocationId', 'MinuteWithinWeek').agg(             
#            percentile_approx('VoicePlusCall', [0.25, 0.5, 0.75], 1).alias('VoicePlusCall')
#        )




#DataThreshOnlyPDF = TrainingSPDF.drop('date').drop('MinuteWithinWeek').drop('week_of_year').toPandas()

# apply the offlineML functions to aggregate thresholds for each antenna
#thresholds = DataThreshOnlyPDF.groupby(['WeeksGroup','LocationId']).var.agg([level1byPL, level2byPL, level3byPL])
#thresholds.reset_index(inplace=True)





print(thresholds.info(verbose=True))

thresholds.to_csv('exports/20220506_ThesholdsStefania_acc_6Thresh.csv', index=False)

print(thresholds.describe())




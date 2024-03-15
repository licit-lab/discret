#spark-submit --driver-memory 100G --executor-cores 10 --conf spark.driver.maxResultSize=15g --conf spark.rapids.sql.enabled=true ExportAlertsAsCsvFiles.py


#AlertsSourceParquetFile = "/rapids/notebooks/LICIT_COMMON_Folder/LICIT_INPUT/notebooks/tools/DetectionParquets/runPaper/"
AlertsSourceParquetFile = "/WORKSPACE/Pierre/Cancan2022/Cancan2022_Paris_detection_OriginalSplit_VoiceSMS_6Thresh/"
#AlertsSourceParquetFile = "/WORKSPACE/Pierre/Cancan2022/Cancan2022_Paris_detection_OriginalSplit_VoiceCall_6Thresh/"
#OutputDetectionsFilePrefix = "DetectionExports/Cancan2022_Lyon_CallSMS_NoTimezone_W"
#OutputDetectionsFilePrefix = "DetectionExports/Cancan2022_Lyon_CallSMS_W"
OutputDetectionsFilePrefix = "DetectionExports/Cancan2022_Paris_CallSMS_W"


AlertsSourceParquetFile = '/WORKSPACE/Pierre/StefaniaResults/Cancan2022_Paris_detection_OriginalSplit_AllMetrics_6Thresh/'
OutputDetectionsFilePrefix = "DetectionExports/Cancan2022_Paris_Stefania_AllMetrics_W"




#AntennasLocInfosFile = '/rapids/notebooks/LICIT_COMMON_Folder/LICIT_INPUT/notebooks/notebooks_pierre/exports/Cancan_Lyon_LocInfos_Summarized.csv'
AntennasLocInfosFile = '/rapids/notebooks/LICIT_COMMON_Folder/LICIT_INPUT/notebooks/notebooks_pierre/exports/Cancan_Paris_LocInfos_Summarized.csv'
#AntennasLocInfosFile = '/WORKSPACE/Pierre/Cancan2022/ClosestND_LocInfos_Summarized.csv'


MinAlertLevel = 1


#import datetime
from datetime import datetime, timedelta, date, timezone
from dateutil import tz

import time

import pyspark
import pyspark.sql.functions as f
from pyspark import SparkContext
from pyspark.sql import *
from pyspark.sql.types import *


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import json

#from shapely.geometry import shape, Point, Polygon

import warnings
#warnings.filterwarnings('ignore')

import sys


spark = SparkSession.builder.appName("ExportDataPierre").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
sc = spark.sparkContext



# found how to navigate between weeks here
# https://stackoverflow.com/questions/304256/whats-the-best-way-to-find-the-inverse-of-datetime-isocalendar

def iso_year_start(iso_year):
    #"The gregorian calendar date of the first day of the given ISO year"
    fourth_jan = datetime(iso_year, 1, 4, 0, 0, 0, tzinfo=tz.gettz("Europe/Paris"))
    delta = timedelta(fourth_jan.isoweekday()-1)
    return fourth_jan - delta 

def iso_to_gregorian(iso_year, iso_week, iso_day):
    #"Gregorian calendar date for the given ISO year, week and day"
    year_start = iso_year_start(iso_year)
    return year_start + timedelta(days=iso_day-1, weeks=iso_week-1)


# framacalc seems to use excel format for dates
# so here's the conversion tool according to: https://www.semicolonworld.com/question/56907/how-to-convert-a-given-ordinal-number-from-excel-to-a-date

def from_excel_ordinal(ordinal, _epoch0=datetime(1899, 12, 31)):
    if ordinal >= 60:
        ordinal -= 1  # Excel leap year bug, 1900 is not a leap year!
    return (_epoch0 + timedelta(days=ordinal)).replace(microsecond=0)






# load source data
DetectionsSparkDF = spark.read.parquet(AlertsSourceParquetFile)

# following lines are specific to Stefania's method (offset on the week calculation, column __index_level_0__)
#DetectionsSparkDF = spark.read.parquet(AlertsSourceParquetFile).drop("__index_level_0__")
#DetectionsSparkDF = DetectionsSparkDF.withColumn('week_of_year', f.col('week_of_year') + f.lit(1))




DetectionsSparkDF.printSchema()



# load the antennas coordinates file
AntennasLocInfosSPDF = spark.read.csv(AntennasLocInfosFile, inferSchema=True, sep=",", header=True)
AntennasLocInfosSPDF = AntennasLocInfosSPDF.drop('NOM_SITE')
AntennasLocInfosSPDF = AntennasLocInfosSPDF.drop('count')
AntennasLocInfosSPDF.printSchema()


# know on which weeks and Location Ids we have performed the detection
#LocIdsList = sorted([x.LocationId for x in DetectionsSparkDF.select('LocationId').distinct().collect()])
#print("found " + str(len(LocIdsList)) + " location groups")
#print(LocIdsList)

#WeekIdsList = sorted([x.week_of_year for x in DetectionsSparkDF.select('week_of_year').distinct().collect()])
WeekIdsList = sorted([x.WeekOfYear for x in DetectionsSparkDF.select('WeekOfYear').distinct().collect()])
print("found " + str(len(WeekIdsList)) + " different weeks")
print(WeekIdsList)



DetectLocDF = DetectionsSparkDF.join(AntennasLocInfosSPDF, on='LocationId', how='inner').withColumnRenamed('LON', 'location_lon').withColumnRenamed('LAT', 'location_lat')
DetectLocDF.printSchema()




# set the timezone, directly in spark

spark.conf.set("spark.sql.session.timeZone", "Europe/Paris")


# set the interval of time for the MinuteWithinWeek values (just the number of minutes within a week)
MinuteStart = 0
MinuteEnd = 60*24*7

# run through all the available weeks and perform an extraction week by week
for week in WeekIdsList:
    
    start_time = time.time()

    #IntervalDF  = DetectLocDF.filter( (DetectLocDF.Anomaly_Level>0) & (DetectLocDF.week_of_year==week) & (DetectLocDF.MinuteWithinWeek>=MinuteStart) & (DetectLocDF.MinuteWithinWeek<MinuteEnd) )
    IntervalDF  = DetectLocDF.filter( (DetectLocDF.Anomaly_Level>0) & (DetectLocDF.WeekOfYear==week) & (DetectLocDF.MinuteWithinWeek>=MinuteStart) & (DetectLocDF.MinuteWithinWeek<MinuteEnd) )
    
    #UnixWeekTime = datetime.combine(iso_to_gregorian(2019,week,1), datetime.min.time(), tzinfo=tz.gettz("Europe/Paris")).timestamp()
    UnixWeekTime = datetime.combine(iso_to_gregorian(2019,week,1), datetime.min.time()).timestamp()
    
    IntervalDF = IntervalDF.withColumn('@timestamp', f.from_unixtime(UnixWeekTime + f.col('MinuteWithinWeek')*60))
    #IntervalDF = IntervalDF.withColumn('@timestamp', f.from_utc_timestamp(f.from_unixtime(UnixWeekTime + f.col('MinuteWithinWeek')*60), "Europe/Paris"))

    PandasOutput = IntervalDF.orderBy(f.col('MinuteWithinWeek').asc()).drop('MinuteWithinWeek').drop('WeekOfYear').toPandas()
    
    PandasOutput.to_csv(OutputDetectionsFilePrefix + str(week) + ".csv", index=False)
    
    print("write CSV file Done in " + str((time.time() - start_time)) + " seconds")

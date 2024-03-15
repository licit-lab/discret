#spark-submit --driver-memory 100G --executor-cores 10 --conf spark.driver.maxResultSize=15g --conf spark.rapids.sql.enabled=true ExportAlertsAsJson.py


AlertsSourceParquetFile = "/WORKSPACE/Pierre/exports/Cancan_Paris_alerts_eric_rel_parquet"
OutputFile = "DetectionExports/RaphaelTestND2NoLvl0.json"



DataAcquisitionYear = 2019


AntennasLocInfosFile = '/rapids/notebooks/LICIT_COMMON_Folder/LICIT_INPUT/notebooks/notebooks_pierre/exports/Cancan_Paris_LocInfos_Summarized.csv'

MinAlertLevel = 1



ExtractDateStart = "2019-04-15T08:00:00"
ExtractDateEnd = "2019-04-16T01:00:00"





#import datetime
from datetime import datetime, timedelta, date, timezone
from dateutil import tz

import time
import json

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

metrics = ['Voice','SMS_3G','PS','CS','Call','SMS_4G','Service_Req','HO']

for m in metrics:
    DetectionsSparkDF = DetectionsSparkDF.drop(m)
DetectionsSparkDF = DetectionsSparkDF.drop('WeeksGroup')
DetectionsSparkDF = DetectionsSparkDF.drop('ALR')


#DetectionsSparkDF = DetectionsSparkDF.withColumn('method', f.lit("Signatures"))

# following lines are specific to Stefania's method (offset on the week calculation, column __index_level_0__)
#DetectionsSparkDF = spark.read.parquet(AlertsSourceParquetFile).drop("__index_level_0__")
#DetectionsSparkDF = DetectionsSparkDF.withColumn('week_of_year', f.col('week_of_year') + f.lit(1))




DetectionsSparkDF.printSchema()



# load the antennas coordinates file
AntennasLocInfosSPDF = spark.read.csv(AntennasLocInfosFile, inferSchema=True, sep=",", header=True)
#AntennasLocInfosSPDF = AntennasLocInfosSPDF.drop('NOM_SITE')
AntennasLocInfosSPDF = AntennasLocInfosSPDF.drop('count')
AntennasLocInfosSPDF.printSchema()


# know on which weeks and Location Ids we have performed the detection
#LocIdsList = sorted([x.LocationId for x in DetectionsSparkDF.select('LocationId').distinct().collect()])
#print("found " + str(len(LocIdsList)) + " location groups")
#print(LocIdsList)

WeekIdsList = sorted([x.week_of_year for x in DetectionsSparkDF.select('week_of_year').distinct().collect()])
print("found " + str(len(WeekIdsList)) + " different weeks")
print(WeekIdsList)



DetectLocDF = DetectionsSparkDF.join(AntennasLocInfosSPDF, on='LocationId', how='inner').withColumnRenamed('LON', 'lon').withColumnRenamed('LAT', 'lat')
DetectLocDF.printSchema()




#DetectLocDF = DetectLocDF.withColumn("location", f.to_json(f.struct("lon", "lat")))


DetectLocDF.printSchema()




# get the date range values from a given event

tdBeginning = datetime.strptime(ExtractDateStart, "%Y-%m-%dT%H:%M:%S")
ScheduleWeekBeginning = tdBeginning.isocalendar()[1]
#print(ScheduleWeekBeginning)
#then the corresponding time and date for minute 0 of this week
FirstWeekDayBeginning = datetime.combine(iso_to_gregorian(DataAcquisitionYear,ScheduleWeekBeginning,1), datetime.min.time())

# Now find the corresponding minute of our event
MinuteWithinWeekBeginning = int(round((tdBeginning-FirstWeekDayBeginning).total_seconds() / 60))
#print(MinuteWithinWeekBeginning)

tdEnding = datetime.strptime(ExtractDateEnd, "%Y-%m-%dT%H:%M:%S")
ScheduleWeekEnding = tdEnding.isocalendar()[1]
#print(ScheduleWeekEnding)
#then the corresponding time and date for minute 0 of this week
FirstWeekDayEnding = datetime.combine(iso_to_gregorian(DataAcquisitionYear,ScheduleWeekEnding,1), datetime.min.time())

# Now find the corresponding minute of our event
MinuteWithinWeekEnding = int(round((tdEnding-FirstWeekDayEnding).total_seconds() / 60))
#print(MinuteWithinWeekEnding)






# build the list of intervals in our time representation
ListTimeIntervals = []

# simplest case: the end week is the same as the beginning week
if ScheduleWeekBeginning == ScheduleWeekEnding:
    # add +1 because we want to include the minute
    ListTimeIntervals.append({'Week':ScheduleWeekBeginning, 'MinutesStart':MinuteWithinWeekBeginning, 'MinutesEnd':MinuteWithinWeekEnding+1})
elif ScheduleWeekBeginning < ScheduleWeekEnding:
    # consecutive weeks, almost as simple
    # first to the end of the first week
    ListTimeIntervals.append({'Week':ScheduleWeekBeginning, 'MinutesStart':MinuteWithinWeekBeginning, 'MinutesEnd':7*24*60})
    
    # if there are weeks in between, just add all the minutes
    for w in range(ScheduleWeekBeginning+1, ScheduleWeekEnding):
        ListTimeIntervals.append({'Week':w, 'MinutesStart':0, 'MinutesEnd':7*24*60})
    
    # then from the beginning of the second one
    ListTimeIntervals.append({'Week':ScheduleWeekEnding, 'MinutesStart':0, 'MinutesEnd':MinuteWithinWeekEnding+1})
else:
    print("there is a mistake, the ending date is anterior to the beginning date")
    
print("intervals we want to extract:")
print(ListTimeIntervals)

TotalTimeExpected = 0
for lti in ListTimeIntervals:
    TotalTimeExpected += lti['MinutesEnd']-lti['MinutesStart']
    
print("Total extraction time: " + str(TotalTimeExpected))






PandasOutputs = []


#exit()



    
start_time = time.time()

# set the timezone, directly in spark

spark.conf.set("spark.sql.session.timeZone", "Europe/Paris")




JsonList = []



# set the interval of time for the MinuteWithinWeek values (just the number of minutes within a week)
for lti in ListTimeIntervals:

    FirstWeekDay = datetime.combine(iso_to_gregorian(DataAcquisitionYear,lti['Week'],1), datetime.min.time(), tzinfo=tz.gettz("Europe/Paris"))

    IntervalDF  = DetectLocDF.filter( (DetectLocDF.Anomaly_Level>=MinAlertLevel) & (DetectLocDF.week_of_year==lti['Week']) & (DetectLocDF.MinuteWithinWeek>=lti['MinutesStart']) & (DetectLocDF.MinuteWithinWeek<lti['MinutesEnd']) )
    
    
    #UnixWeekTime = datetime.combine(iso_to_gregorian(2019,lti['Week'],1), datetime.min.time(), tzinfo=tz.gettz("Europe/Paris")).timestamp()
    
    #IntervalDF = IntervalDF.withColumn('@timestamp', f.from_unixtime(UnixWeekTime + f.col('MinuteWithinWeek')*60))

    #PandasOutput = IntervalDF.orderBy(f.col('MinuteWithinWeek').asc()).drop('MinuteWithinWeek').drop('week_of_year').toPandas()
    PandasOutput = IntervalDF.orderBy(f.col('MinuteWithinWeek').asc()).drop('week_of_year').toPandas()
    
    for index, row in PandasOutput.iterrows():
        CurrentTime = FirstWeekDay + timedelta(minutes=row['MinuteWithinWeek'])
        JsonList.append( { '@timestamp':CurrentTime.isoformat(),
                           'method': 'Signature',
                           'location': {'lon':row['lon'], 'lat':row['lat']},
                           'LocName': row['NOM_SITE'],
                           'LocId': row['LocationId'],
                           'Anomaly_Level': row['Anomaly_Level'],
                           'Anomaly_Text': 'level_' + str(row['Anomaly_Level'])
                         })
    
    #PandasOutputs.append(PandasOutput)
    
    #PandasOutput.to_csv(OutputDetectionsFilePrefix + str(week) + ".csv", index=False)
    
    #print("write CSV file Done in " + str((time.time() - start_time)) + " seconds")

    
    
#WholeDF = pd.concat(PandasOutputs)

#print(WholeDF.info(verbose=True))

    
#JsonList = WholeDF.to_dict(orient='records')


with open(OutputFile, 'w', encoding='utf8') as json_file:
    json.dump(JsonList, json_file, ensure_ascii=False)
    
    
print("write JSON file Done in " + str((time.time() - start_time)) + " seconds")

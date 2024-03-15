import pandas as pd
import numpy as np
import datetime
import time

import pyspark
import pyspark.sql.functions as f
from pyspark import SparkContext
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import weekofyear
from pyspark.sql.functions import expr


import warnings
warnings.filterwarnings('ignore')



#nohup spark-submit --driver-memory 100G --executor-cores 10 --conf spark.driver.maxResultSize=15g --conf spark.rapids.sql.enabled=true 20220208_RegenerateCancanWorkingData.py &


spark = SparkSession.builder.appName("GenerateDataPierre").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
sc = spark.sparkContext





def iso_year_start(iso_year):
    #"The gregorian calendar date of the first day of the given ISO year"
    fourth_jan = datetime.date(iso_year, 1, 4)
    delta = datetime.timedelta(fourth_jan.isoweekday()-1)
    return fourth_jan - delta 

def iso_to_gregorian(iso_year, iso_week, iso_day):
    #"Gregorian calendar date for the given ISO year, week and day"
    year_start = iso_year_start(iso_year)
    return year_start + datetime.timedelta(days=iso_day-1, weeks=iso_week-1)





start_time = time.time()




# some constants that will be used along the way...

LocationPrefix = "Lyon"
DatasetPrefix = "Cancan"
LocationsSubsetSize = 100


# coords are used to group antennas between them
coords = ['COORD_X','LAT','COORD_Y','LON']

# those are the metadata used to sum the number of requests
meta = ['LocationId', 'time_utc', 'time_local']

# those are the metrics once we have gathered antennas and stuff
metrics = ['Voice','SMS_3G','PS','CS','Call','SMS_4G','Service_Req','HO']



# loading the list of antennas that correspond to the Paris region
# we might have to update it someday but for now i'm using an old file used previously by Evelyne
#topo_df = pd.read_csv('antennas_selection/antennas_topo_ClosestND.csv')
topo_df = pd.read_csv('antennas_selection/antennas_topo_Lyon.csv')
# this line is specific to the previous file, due to an unnecessary column
#topo_df = topo_df.drop('Unnamed: 0', axis=1)
#topo_df = topo_df.drop('field_1', axis=1)







# loading all data from the raw files

#3G
PATH_3G = '/DISCRET_SOURCE_DATAs/TestAngelo/DIOD/451FR/3G/*/*/*.parquet'
data3g = spark.read.parquet(PATH_3G, inferSchema=True, header=True)

# drop the unnecessary columns (at least the ones deemed unnecessary)
data3g = data3g.drop('Video') \
    .drop('Data_PS') \
    .drop('Other_CS') \
    .drop('SS')
data3g.printSchema()

loc3G = topo_df['LocInfo'].loc[topo_df.TECHNO=='3G'].values.tolist()
# just modified this line to adapt it to the processing part
cells_3g = data3g.filter(f.col('LocInfo').isin(loc3G)).withColumnRenamed('SMS', 'SMS_3G')



# do just the same for 4G

#4G
PATH_4G = '/DISCRET_SOURCE_DATAs/TestAngelo/DIOD/451FR/4G/*/*/*.parquet'
data4g = spark.read.parquet(PATH_4G, inferSchema=True, header=True)

# drop what's not useful
data4g = data4g.drop('Add_Bearer_Mod') \
    .drop('Ext_Service_Req')
data4g.printSchema()

loc4G = topo_df['LocInfo'].loc[topo_df.TECHNO=='4G'].values.tolist()
# just modified this line to adapt it to the processing part
# don't forget to differentiate SMS from 3G to 4G services
cells_4g = data4g.filter(f.col('LocInfo').isin(loc4G)).withColumnRenamed('SMS', 'SMS_4G')






# now reproducing the data formatting steps used by Evelyne

# Agregate by location & date
metrics_3g = ['Voice','SMS_3G','PS','CS']
metrics_4g = ['Call','SMS_4G','Service_Req','HO']

cells_3g = cells_3g.groupBy(['LocInfo','Ts']).sum().drop('sum(Ts)')
cells_4g = cells_4g.groupBy(['LocInfo','Ts']).sum().drop('sum(Ts)')

for m in metrics_3g:
    cells_3g = cells_3g.withColumnRenamed('sum('+m+')', m)
for m in metrics_4g:
    cells_4g = cells_4g.withColumnRenamed('sum('+m+')', m)





# get the whole list of 3g and 4g cells that are within the area of interest
TOPO_FILEPATH = '/DISCRET_SOURCE_DATAs/TestAngelo/DIOD/CELLS_TOPO/LocInfo_cancan_allcells_names_wgs.csv'
topo = spark.read.csv(TOPO_FILEPATH, inferSchema=True, sep=";", header=True)

# select only the ones we want in both 4g and 3g configurations
topo_3g = topo.filter(f.col('LocInfo').isin(loc3G))
topo_4g = topo.filter(f.col('LocInfo').isin(loc4G))

# we just want to generate IDs to the antennas we will eventually keep
topo_3g = topo_3g.drop('TECHNO').drop('LAC').drop('min_dt').drop('max_dt').drop('NOM_SITE').drop('CI').withColumnRenamed('LocInfo', 'LocInfo_3g')
topo_4g = topo_4g.drop('TECHNO').drop('LAC').drop('min_dt').drop('max_dt').drop('NOM_SITE').drop('CI').withColumnRenamed('LocInfo', 'LocInfo_4g')


# join them
topo_3g = topo_3g.alias('topo_3g')
topo_4g = topo_4g.alias('topo_4g')
topo_join = topo_3g.join(topo_4g, on=coords, how='outer')


# now what we want is to generate an ID for each [coords] set of values
# way to perform that: sort by coordinates then assign the rank as an id
topo_join = topo_join.withColumn("LocationId", f.dense_rank().over(Window.orderBy(coords)))

# verify how many antennas are kept after aggregation 
#print(topo_join.agg({"id": "max"}).collect()[0])


antennasNumberDf = topo_join.agg({"LocationId": "max"}).collect()[0]

AntennasNumber = int( antennasNumberDf['max(LocationId)'] )

print("found " + str(AntennasNumber) + " antennas")


# store this association as this will be used as a record to know the antenna coords
#topo_join.toPandas().to_csv('/WORKSPACE/Pierre/Cancan2022_verif/ClosestND_LocInfo_Ids.csv', index=False)
topo_join.toPandas().to_csv('/WORKSPACE/Pierre/Cancan2022/Lyon_LocInfo_Ids.csv', index=False)









#spark.conf.set("spark.sql.session.timeZone", "Europe/Paris")







# now assign the antennas id associations to the raw data

# first drop the now useless informations so that they are not joined

topo_NoCoords = topo_join.drop('COORD_X').drop('LAT').drop('COORD_Y').drop('LON')

# remove all the ambiguities
topo_NoCoords_3g = topo_NoCoords.select('LocInfo_3g','LocationId').distinct()
topo_NoCoords_4g = topo_NoCoords.select('LocInfo_4g','LocationId').distinct()

# add the location ID stuff
c3 = cells_3g.alias('c3')
topo_NoCoords_3g = topo_NoCoords_3g.alias('topo_NoCoords_3g')
c3_topo = c3.join(topo_NoCoords_3g, c3.LocInfo == topo_NoCoords_3g.LocInfo_3g).drop('LocInfo_3g')

c4 = cells_4g.alias('c4')
topo_NoCoords_4g = topo_NoCoords_4g.alias('topo_NoCoords_4g')
c4_topo = c4.join(topo_NoCoords_4g, c4.LocInfo == topo_NoCoords_4g.LocInfo_4g).drop('LocInfo_4g')


# this is important to ensure that time_utc is set to actual time utc
spark.conf.set("spark.sql.session.timeZone", "Etc/Universal") 



# Reformat date
c3_topo = c3_topo.na.fill(0) \
    .withColumn('time_utc', f.from_unixtime('Ts')) \
    .withColumn('time_local', f.from_utc_timestamp('time_utc', 'Europe/Paris')) \
    .drop('Ts')

c4_topo = c4_topo.na.fill(0) \
    .withColumn('Ts', f.col('Ts')*60) \
    .withColumn('time_utc', f.from_unixtime('Ts')) \
    .withColumn('time_local', f.from_utc_timestamp('time_utc', 'Europe/Paris')) \
    .drop('Ts')




# old version by Evelyne below

# Reformat date
#c3_topo = c3_topo.na.fill(0) \
#    .withColumn('Ts', f.col('Ts').cast('string')) \
#    .withColumn('time_utc', f.from_unixtime('Ts', 'yyyy-MM-dd HH:mm')) \
#    .withColumn('time_local', f.from_utc_timestamp('time_utc', 'Europe/Paris')) \
#    .drop('Ts')

#c4_topo = c4_topo.na.fill(0) \
#    .withColumn('Ts', f.col('Ts')*60) \
#    .withColumn('Ts', f.col('Ts').cast('string')) \
#    .withColumn('time_utc', f.from_unixtime('Ts', 'yyyy-MM-dd HH:mm')) \
#    .withColumn('time_local', f.from_utc_timestamp('time_utc', 'Europe/Paris')) \
#    .drop('Ts')





# Reformat date
#c3_topo = c3_topo.na.fill(0) \
#    .withColumn('Ts', f.from_unixtime('Ts')) \
#    .withColumn('time_utc', f.from_utc_timestamp('Ts', 'Etc/Universal')) \
#    .withColumn('time_local', f.from_utc_timestamp('Ts', 'Europe/Paris')) \
#    .drop('Ts')

#c4_topo = c4_topo.na.fill(0) \
#    .withColumn('Ts', f.col('Ts')*60) \
#    .withColumn('Ts', f.from_unixtime('Ts')) \
#    .withColumn('time_utc', f.from_utc_timestamp('Ts', 'Etc/Universal')) \
#    .withColumn('time_local', f.from_utc_timestamp('Ts', 'Europe/Paris')) \
#    .drop('Ts')


#    .orderBy('time_utc')
#    .withColumn('Ts', f.col('Ts').cast('string')) \
#    .withColumn('Date', f.from_unixtime(f.col('Ts'), 'yyyy-MM-dd HH:mm')) \
#    .orderBy('Date') \
#    .drop('Ts')






# Timezone conversion (NEW)
#cells_data = cells_data.na.fill(0) \
#    .withColumn('start_min', f.col('start_min').cast('string')) \
#    .withColumn('time_utc', f.from_unixtime('start_min', 'yyyy-MM-dd HH:mm')) \
#    .withColumn('time_local', f.from_utc_timestamp('time_utc', 'Europe/Paris')) \
#    .drop('start_min') \
#    .orderBy('time_utc')



# Agregate antennas by LocationId + date
clusters3 = c3_topo.groupBy(meta).sum().drop('sum(LocationId)')
for m in metrics_3g:
    clusters3 = clusters3.withColumnRenamed('sum('+m+')', m)

clusters4 = c4_topo.groupBy(meta).sum().drop('sum(LocationId)')
for m in metrics_4g:
    clusters4 = clusters4.withColumnRenamed('sum('+m+')', m)

    

# Reshape
# not completely sure the orderBy thing is really important but still...
#clusters3 = clusters3.orderBy('Date')
#clusters4 = clusters4.orderBy('Date')



# finally merge 3G and 4G data
c3 = clusters3.alias('c3')
c4 = clusters4.alias('c4')
clusters = c3.join(c4, on=meta, how='outer')#.orderBy('Date')


clusters = clusters.withColumn('WeekOfYear', weekofyear(clusters.time_local))
clusters = clusters.withColumn('MinuteWithinWeek', datediff("time_local", date_trunc('week', "time_local"))*60*24 + 60*hour("time_local") + minute("time_local"))
clusters = clusters.withColumn('WeeksGroup', expr("WeekOfYear % 3"))



#clusters = clusters.withColumn('WeekOfYear', weekofyear(clusters.time_utc))
#clusters = clusters.withColumn('MinuteWithinWeek', datediff("time_utc", date_trunc('week', "time_utc"))*60*24 + 60*hour("time_utc") + minute("time_utc"))
#clusters = clusters.withColumn('WeeksGroup', expr("WeekOfYear % 3"))
#clusters.show(10)


#df.withColumn("input_timestamp",
#    to_timestamp(col("input_timestamp")))
#  .withColumn("week_of_year", date_format(col("input_timestamp"), "w"))
#  .show(false)

print("All operations done in " + str((time.time() - start_time)) + " seconds until printSchema()")

clusters.printSchema()



#exit()

OutputParquetFilesLoc = '/WORKSPACE/Pierre/Cancan2022/Cancan2022_Lyon'

start_time = time.time()

try:
    clusters.repartition("WeekOfYear", "LocationId").sortWithinPartitions("MinuteWithinWeek").write.partitionBy("WeekOfYear", "LocationId").option('header', 'true').mode("append").parquet(OutputParquetFilesLoc)
    
    print("write parquet Done in " + str((time.time() - start_time)) + " seconds")
except:
    print("=============")
    print("ERROR while trying to write paquet " + exportFileName)
    print("=============")





exit()


# alright so now just store the mess, week by week

#clusters = clusters.withColumn('Date', f.to_timestamp(f.col('Date'), 'yyyy-MM-dd HH:mm'))

# found how to navigate between weeks here
# https://stackoverflow.com/questions/304256/whats-the-best-way-to-find-the-inverse-of-datetime-isocalendar

# separating weeks within the dataset to make the training and testing tasks easier
for AntennasSubsetId in range(LocationsSubsetSize, AntennasNumber, LocationsSubsetSize):
    
    print("working on antennas within range " + str(AntennasSubsetId) + " - " + str(AntennasSubsetId+LocationsSubsetSize))
    
    #AntennasSubsetCluster = clusters.filter((clusters.LocationId >= AntennasSubsetId) & (clusters.LocationId < AntennasSubsetId+LocationsSubsetSize))


    for wId in range(11,25):
        start_time = time.time()
        
        IntervIn = iso_to_gregorian(2019,wId,1)
        IntervOut = iso_to_gregorian(2019,wId+1,1)

        print("storing dates between " + str(IntervIn) + " and " + str(IntervOut) + " as week " + str(wId))

        storing = clusters.filter((clusters.LocationId >= AntennasSubsetId) & (clusters.LocationId < AntennasSubsetId+LocationsSubsetSize) & (clusters.Date >= IntervIn) & (clusters.Date < IntervOut))
        
        exportFileName = '/WORKSPACE/Pierre/exports/dataset_' + DatasetPrefix + '/' + LocationPrefix + '_week' + str(wId) + '_subset_' + str(AntennasSubsetId) + '.parquet'
        
        print("file to be created: " + exportFileName)
        
        start_time = time.time()
        
        try:
            storing.write.mode("overwrite").parquet(exportFileName)
                
            print("write parquet Done in " + str((time.time() - start_time)) + " seconds")
        except:
            print("=============")
            print("ERROR while trying to write paquet " + exportFileName)
            print("=============")
        

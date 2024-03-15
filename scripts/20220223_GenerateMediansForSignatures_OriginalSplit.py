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


import warnings
warnings.filterwarnings('ignore')


import sys



#nohup spark-submit --driver-memory 100G --executor-cores 10 --conf spark.driver.maxResultSize=15g --conf spark.rapids.sql.enabled=true 20220223_GenerateMediansForSignatures_OriginalSplit.py &


spark = SparkSession.builder.appName("GenerateMediansAnritsu").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
sc = spark.sparkContext




# some constants


#SourceParquetFilesLoc = '/WORKSPACE/Pierre/Cancan2022_verif/ClosestND_Data/'
#OutputParquetFilesLoc = '/WORKSPACE/Pierre/Cancan2022_verif/ClosestND_meds/'

SourceParquetFilesLoc = '/WORKSPACE/Pierre/Cancan2022/Cancan2022_Lyon/'
OutputParquetFilesLoc = '/WORKSPACE/Pierre/Cancan2022/Cancan2022_Lyon_meds_OriginalSplit_fillna/'


#NodesNumber = 100
#NodeIdStart = int(sys.argv[1])
#print('start with id ' + str(NodeIdStart*NodesNumber))


# those are the metadata used to sum the number of requests
meta = ['LocationId', 'WeekOfYear', 'MinuteWithinWeek']

drop = ['time_local', 'time_utc', 'WeeksGroup']

# those are the metrics once we have gathered antennas and stuff
metrics = ['Voice','SMS_3G','PS','CS','Call','SMS_4G','Service_Req','HO','VoicePlusCall']





start_time = time.time()

# read the files generated 
SparkData = spark.read.parquet(SourceParquetFilesLoc, header=True)
SparkData.printSchema()

print("read parquet Done in " + str((time.time() - start_time)) + " seconds")

#SparkData = SparkData.filter( (SparkData.LocationId >= NodesNumber * NodeIdStart ) & (SparkData.LocationId < NodesNumber * (NodeIdStart+1)) )

for d in drop:
    SparkData = SparkData.drop(d)


# generate the sum column on the fly
#SparkData = SparkData.withColumn( "VoicePlusCall", ( f.nanvl(f.col("Voice"), f.lit(0)) + f.nanvl(f.col("Call"), f.lit(0))) )
SparkData = SparkData.withColumn( "VoicePlusCall", (f.nanvl(SparkData.Voice, f.lit(0)) + f.nanvl(SparkData.Call, f.lit(0))).cast('long') ).fillna(value=0, subset=['VoicePlusCall'])

SparkData.printSchema()



lGroups = []
for i in range(0,3):
    lGroups.append([i])

dfGroups = sc.parallelize(lGroups).toDF(["WeeksGroup"])
dfGroups = dfGroups.withColumn("WeeksGroup", dfGroups.WeeksGroup.cast('integer'))
#dfGroups.show()



#lMinutes = []
#for i in range(0,7*24*60):
#    lMinutes.append([i])
#dfMinutes = sc.parallelize(lMinutes).toDF(["MinuteWithinWeek"])
#dfMinutes = dfMinutes.withColumn("MinuteWithinWeek", dfMinutes.MinuteWithinWeek.cast('integer'))


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






SparkData.printSchema()





lGroups = []
for i in range(0,3):
    lGroups.append([i])

dfGroups = sc.parallelize(lGroups).toDF(["TrainingWeeksGroup"])
dfGroups = dfGroups.withColumn("TrainingWeeksGroup", dfGroups.TrainingWeeksGroup.cast('integer'))




TrainWeeks.show()








JoinData = SparkData.crossJoin(TrainWeeks)

JoinData.printSchema()






start_time = time.time()

# keep only the weeks that correspond to 

TrainingSPDF = JoinData.filter( JoinData.WeekOfYear==JoinData.TrainWeek).repartition("WeeksGroup", "LocationId")






TrainingSPDF.printSchema()

#print("join Done in " + str((time.time() - start_time)) + " seconds")


start_time = time.time()



MedianSpark = TrainingSPDF.groupBy('WeeksGroup', 'LocationId', 'MinuteWithinWeek').agg( 
            percentile_approx('Voice', [0.25, 0.5, 0.75], 1).alias('Voice'),
            percentile_approx('SMS_3G',[0.25, 0.5, 0.75], 1).alias('SMS_3G'),
            percentile_approx('PS', [0.25, 0.5, 0.75], 1).alias('PS'),
            percentile_approx('CS', [0.25, 0.5, 0.75], 1).alias('CS'),
            percentile_approx('Call', [0.25, 0.5, 0.75], 1).alias('Call'),
            percentile_approx('SMS_4G', [0.25, 0.5, 0.75], 1).alias('SMS_4G'),
            percentile_approx('Service_Req', [0.25, 0.5, 0.75], 1).alias('Service_Req'),
            percentile_approx('HO', [0.25, 0.5, 0.75], 1).alias('HO'),
            percentile_approx('VoicePlusCall', [0.25, 0.5, 0.75], 1).alias('VoicePlusCall')
        )



MedianSpark = MedianSpark.na.fill(0)

MedianSpark.printSchema()



MedianSpark.repartition("WeeksGroup", "LocationId").sortWithinPartitions("MinuteWithinWeek").write.partitionBy("WeeksGroup", "LocationId").option('header', 'true').mode("overwrite").parquet(OutputParquetFilesLoc)


print("write parquet Done in " + str((time.time() - start_time)) + " seconds")


#MediansDFSP.show(10)




# now adding the missing 0s...
# first get the list of different weekgroups and locIds
#WeekLocGroups = MedianSpark.groupBy("WeeksGroup", "LocationId").agg({'MinuteWithinWeek':'count'}).drop('count(MinuteWithinWeek)')

# then multiply by the number of possible minutes
#MinWeekLocGroups = WeekLocGroups.crossJoin(dfMinutes)


# do the join to add missing minutes with null values...
#MedianSpark = MedianSpark.join(MinWeekLocGroups, ['WeeksGroup','LocationId','MinuteWithinWeek'], how='outer')

# ... then fill with 0s
#MedianSpark = MedianSpark.na.fill(0)

# finally save with a repartition THEN a sort within partitions
#MedianSpark.repartition("WeeksGroup", "LocationId").sortWithinPartitions("MinuteWithinWeek").write.partitionBy("WeeksGroup", "LocationId").option('header', 'true').mode("append").parquet(OutputParquetFilesLoc)
#MedianSpark.write.partitionBy("WeeksGroup", "LocationId").option('header', 'true').mode("append").parquet(OutputParquetFilesLoc)



#AntennaGroupSize = 30


#for AntennaId in range(1, 1280, AntennaGroupSize):
    
#    print("computing Antennas " + str(AntennaId) + " to " + str(AntennaId+AntennaGroupSize-1))

#    # compute the signatures, week by week
#    for week in range(MinWeek, MaxWeek+1):
#    #for week in range(MinWeek, 12+1):
#        start_time = time.time()

#        print("computing week " + str(week))

#        FilteredSpark = SparkData.filter((SparkData.week_of_year!=week) & (SparkData.LocationId>=AntennaId) & (SparkData.LocationId<AntennaId+AntennaGroupSize))
        #FilteredSpark = SparkData.filter(SparkData.week_of_year!=week)

#        MedianSpark = FilteredSpark.groupBy('MinuteWithinWeek', 'LocationId').agg( 
#            percentile_approx('Voice', 0.5, 1).alias('Voice'),
#            percentile_approx('SMS_3G', 0.5, 1).alias('SMS_3G'),
#            percentile_approx('PS', 0.5, 1).alias('PS'),
#            percentile_approx('CS', 0.5, 1).alias('CS'),
#            percentile_approx('Call', 0.5, 1).alias('Call'),
#            percentile_approx('SMS_4G', 0.5, 1).alias('SMS_4G'),
#            percentile_approx('Service_Req', 0.5, 1).alias('Service_Req'),
#            percentile_approx('HO', 0.5, 1).alias('HO')
#        ).withColumn('Week', lit(week)).orderBy('MinuteWithinWeek')
#        #MedianSpark.printSchema()
#
#        #exportFileName = OutputParquetFilesLoc + "_w" + str(week)
#        exportFileName = OutputParquetFilesLoc

#        try:
#            MedianSpark.write.partitionBy("Week", "LocationId").mode("append").parquet(exportFileName)

#            print("write parquet Done in " + str((time.time() - start_time)) + " seconds")
#        except:
#            print("=============")
#            print("ERROR while trying to write paquet " + exportFileName)
#            print("=============")

        #MedianSpark.show(30)


    #break





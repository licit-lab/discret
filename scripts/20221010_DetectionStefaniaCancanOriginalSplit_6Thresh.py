import datetime

import time

import pyspark
import pyspark.sql.functions as f
from pyspark import SparkContext
from pyspark.sql import *
from pyspark.sql.types import *

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


import sys



#nohup spark-submit --driver-memory 100G --executor-cores 10 --conf spark.driver.maxResultSize=15g --conf spark.rapids.sql.enabled=true 20221010_DetectionStefaniaCancanOriginalSplit_6Thresh.py &

spark = SparkSession.builder.appName("DetectionCancan2022OriginalSplit").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
sc = spark.sparkContext



# those are the metadata used to sum the number of requests
meta = ['WeeksGroup', 'LocationId', 'MinuteWithinWeek']

# those are the metrics once we have gathered antennas and stuff
MetricsFilter = ['Voice','SMS_3G','Call','SMS_4G']
metrics = ['SMS_3G','SMS_4G']

SourceParquetFilesLoc = '/WORKSPACE/Pierre/Cancan2022/Cancan2022_Lyon/'
ErrorsParquetFileLoc = '/WORKSPACE/Pierre/StefaniaResults/Adaptive_Errors4Services'


ParquetFilesDistribsLoc = '/WORKSPACE/Pierre/StefaniaResults/Cancan_Paris_distribs_OriginalSplit_SMS3G4G'
CsvFilesThresholdsLoc = '/WORKSPACE/Pierre/StefaniaResults/Cancan2022_Paris_Thresholds_OriginalSplit_SMS3G4G_6Thresh.csv'

ParquetDetectionLoc = '/WORKSPACE/Pierre/StefaniaResults/Cancan2022_Paris_detection_OriginalSplit_SMS3G4G_6Thresh/'




print(ParquetFilesDistribsLoc)
print(CsvFilesThresholdsLoc)
print(ParquetDetectionLoc)



# load the data
ErrorsDFSP = spark.read.parquet(ErrorsParquetFileLoc)
ErrorsDFSP.printSchema()



#OriginalDataDFSP = spark.read.parquet(SourceParquetFilesLoc).drop('WeeksGroup')
OriginalDataDFSP = spark.read.parquet(SourceParquetFilesLoc)

# generate the sum column on the fly
#OriginalDataDFSP = OriginalDataDFSP.withColumn( "VoicePlusCall", (f.nanvl(OriginalDataDFSP.Voice, f.lit(0)) + f.nanvl(OriginalDataDFSP.Call, f.lit(0))).cast('long') ).fillna(value=0, subset=['VoicePlusCall'])
OriginalDataDFSP.printSchema()



#OriginalDataDFSP = spark.read.parquet(SourceParquetFilesLoc).drop('WeeksGroup')
DistribsDFSP = spark.read.parquet(ParquetFilesDistribsLoc)

# generate the sum column on the fly
#OriginalDataDFSP = OriginalDataDFSP.withColumn( "VoicePlusCall", (f.nanvl(OriginalDataDFSP.Voice, f.lit(0)) + f.nanvl(OriginalDataDFSP.Call, f.lit(0))).cast('long') ).fillna(value=0, subset=['VoicePlusCall'])
DistribsDFSP.printSchema()


ThresholdsDF = pd.read_csv(CsvFilesThresholdsLoc)
print(ThresholdsDF.info(verbose=True))











LocIdsList = sorted([x.LocationId for x in ErrorsDFSP.select('LocationId').distinct().collect()])

print("found " + str(len(LocIdsList)) + " location groups")




WeeksList = sorted([x.WeekOfYear for x in ErrorsDFSP.select('WeekOfYear').distinct().collect()])

print("found " + str(len(WeeksList)) + " weeks")
print(WeeksList)



TestWeeksGroupsList = [ [12,13,14,15], [16,17,18,19], [20,21,22,23,24] ]

TrainWeeksGroupsList = []

for i in range(0,len(TestWeeksGroupsList)):
    CurrList = []
    for j in range(0,len(TestWeeksGroupsList)):
        if i!=j:
            CurrList += TestWeeksGroupsList[j]
    TrainWeeksGroupsList.append(CurrList)

print("Training Weeks Ids:")
print(TrainWeeksGroupsList)



AllWeeksGroupsList = range(0,3)







from tqdm import tqdm

import scipy.stats as stats




MinALSThreshold = 1 / (60*24*365.25*10)

def compute_alr_eric_local(df:pd.DataFrame, distrib:dict, metrics:list) -> pd.DataFrame:
    """Computes the Anomaly Likelihood Rate (ALR) over the input dataframe.
    The ALR corresponds to the sum of the logs of the p-value for each service data.
    The p-value is obtained for each service data by fitting a Gamma distribution over the dataset.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The input dataframe (usually containing AE values)
    distrib: dict
        The error distribution parameters for all services
    metrics: list
        The list of column names corresponding to the service data

    Returns
    -------
    df: pandas.core.frame.DataFrame
        The same dataframe with the additional ALR column
    """

    res = pd.DataFrame().reindex_like(df)
    als = pd.DataFrame(index = res.index, columns = metrics)
    
    CopyCols = list(set(df.columns) - set(metrics))
    res[CopyCols] = df[CopyCols].copy(deep=True)

    for m in metrics:
        mLoc = df.columns.get_loc(m)
            
        SeparationThresh = distrib.loc['thresh',m]
        proba = distrib.loc['proba',m]
        nvalues = distrib.loc['nvalues',m]
        arg = distrib.loc['k',m]
        loc = distrib.loc['loc',m]
        scale = distrib.loc['theta',m]

        #try:
        if nvalues<3:
            #print("case where the gamma law was not fitted on metric " + m)
            res.loc[:,m] = pd.Series(np.nan, index=df.index)
        else:
            # default value is 1-proba
            res.loc[:,m] = pd.Series((1. - proba), index=df.index)
            # the gamma law was fitted on this metric
            indexThresh = df.index[df[m]>SeparationThresh]

            res.loc[indexThresh,m] = (1. - proba) * stats.gamma.sf(df.loc[indexThresh, m], arg, loc=loc, scale=scale)
            res.loc[indexThresh,m].clip(lower=MinALSThreshold, inplace=True)


        als[m] = pd.Series(np.log(res[m]), index=res.index)
            
        #     break
        #except:
        #    print("issue with this block")
        #    print(nvalues.info(verbose=True))

    res['ALR'] = als.sum(axis=1)
    return res





def set_levels_local(df, thresh):
    """Set anomaly levels on the input data according to the ALR value at each timestamp.
    The data can be labelled as: normal, pre-alert (level 1), alert (level 2), maximal anomaly (level 3).

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The input dataframe with ALR values
    thresh: pandas.core.frame.Series
        The ALR thresholds of the dataframe for each anomaly level

    Returns
    -------
    df: pandas.core.frame.DataFrame
        The dataframe with anomaly levels (0 to 3)
    """

    df['Anomaly_Level'] = 0
    df.loc[df.error > thresh['level1byPL'], 'Anomaly_Level'] = 1
    df.loc[df.error > thresh['level2byPL'], 'Anomaly_Level'] = 2
    df.loc[df.error > thresh['level3byPL'], 'Anomaly_Level'] = 3
    return df



  
def set_levels_local_6Thresh(df, thresh):
    """Set anomaly levels on the input data according to the ALR value at each timestamp.
    The data can be labelled as: normal, pre-alert (level 1), alert (level 2), maximal anomaly (level 3).

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The input dataframe with ALR values
    thresh: pandas.core.frame.Series
        The ALR thresholds of the dataframe for each anomaly level

    Returns
    -------
    df: pandas.core.frame.DataFrame
        The dataframe with anomaly levels (0 to 3)
    """

    df['Anomaly_Level'] = 0
    df.loc[df.ALR < thresh['level1byPL6Thresh'], 'Anomaly_Level'] = 1
    df.loc[df.ALR < thresh['level2byPL6Thresh'], 'Anomaly_Level'] = 2
    df.loc[df.ALR < thresh['level3byPL6Thresh'], 'Anomaly_Level'] = 3
    df.loc[df.ALR < thresh['level4byPL6Thresh'], 'Anomaly_Level'] = 4
    df.loc[df.ALR < thresh['level5byPL6Thresh'], 'Anomaly_Level'] = 5
    df.loc[df.ALR < thresh['level6byPL6Thresh'], 'Anomaly_Level'] = 6
    return df








totalAnomalies = []



with tqdm(total=len(LocIdsList)*len(AllWeeksGroupsList), desc='ALR computing') as pbar:
    # run 2 imbricated loops, this hasn't much consequence on the output thanks to the filter / partitioning thing 
    for WGrp in AllWeeksGroupsList:

        TestWeeksIdsList = TestWeeksGroupsList[WGrp]
        
        for LocId in LocIdsList:
            
            #if LocId>20:
            #    break


            ErrorsDFsList = []
            for wk in TestWeeksIdsList:
                wkDF = ErrorsDFSP.filter((ErrorsDFSP.WeekOfYear==wk) & (ErrorsDFSP.LocationId == LocId)).drop('WeekOfYear').toPandas().set_index('MinuteWithinWeek')
                wkDF = wkDF.reindex(range(0,24*7*60), fill_value=0).assign(WeekOfYear=wk, LocationId=LocId, WeeksGroup=WGrp).fillna(0).reset_index()

                ErrorsDFsList.append( wkDF )

            VarErrors = pd.concat(ErrorsDFsList)
            #print(VarErrors.info(verbose=True))
            #print(VarErrors.describe())
            
            
            GammaLawDF = DistribsDFSP.filter((DistribsDFSP.LocationId==LocId) & (DistribsDFSP.WeeksGroup==WGrp)).toPandas()
            GammaLawDF.set_index('GammaParam', inplace=True)
            #print(GammaLawDF.info(verbose=True))
            #print(GammaLawDF.describe())
            #print(GammaLawDF)


            AnomalyLevel = compute_alr_eric_local(VarErrors, GammaLawDF, metrics)
            

            thresholds = ThresholdsDF.loc[(ThresholdsDF['LocationId']==LocId) & (ThresholdsDF['WeeksGroup']==WGrp)]
            #print("==== thresholds DF ====")
            #print(thresholds.info(verbose=True))
            #print(thresholds.describe())
            #print(thresholds)
            if len(thresholds)==0:
                pbar.update(1)
                continue

            ThreshDict = thresholds.to_dict(orient='records')[0]
            #print(ThreshDict)
            #break

            # this one should not be used anymore actually
            #replace_inf_local(AnomalyLevel, metrics, ThreshDict)

            set_levels_local_6Thresh(AnomalyLevel, ThreshDict)
            #AnomalyLevel = AnomalyLevel.fillna(0)
            #print("==== anomaly levels DF ====")
            #print(AnomalyLevel.info(verbose=True))
            #print(AnomalyLevel.describe())
            #print(AnomalyLevel)
            
            AnomalyLevel = AnomalyLevel.drop(columns=metrics)

            totalAnomalies.append(AnomalyLevel)

            pbar.update(1)
        #break
    #break

totalAnomDF = pd.concat(totalAnomalies)

#totalAnomDF.info(verbose=True)



totalAnomDF.to_parquet(path=ParquetDetectionLoc, partition_cols=['WeekOfYear'], index=False)





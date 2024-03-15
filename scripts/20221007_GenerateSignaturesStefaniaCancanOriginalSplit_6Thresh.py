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



#nohup spark-submit --driver-memory 100G --executor-cores 10 --conf spark.driver.maxResultSize=15g --conf spark.rapids.sql.enabled=true 20221007_GenerateSignaturesStefaniaCancanOriginalSplit_6Thresh.py &

spark = SparkSession.builder.appName("GenerateSignaturesCancan2022OriginalSplit").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
sc = spark.sparkContext



# those are the metadata used to sum the number of requests
meta = ['WeeksGroup', 'LocationId', 'MinuteWithinWeek']

# those are the metrics once we have gathered antennas and stuff
MetricsFilter = ['Voice','SMS_3G','Call','SMS_4G']
#metrics = ['Voice','SMS_3G','Call','SMS_4G']
metrics = ['SMS_3G','SMS_4G']

SourceParquetFilesLoc = '/WORKSPACE/Pierre/Cancan2022/Cancan2022_Lyon/'
ErrorsParquetFileLoc = '/WORKSPACE/Pierre/StefaniaResults/Adaptive_Errors4Services'


ParquetFilesDistribsLoc = '/WORKSPACE/Pierre/StefaniaResults/Cancan_Paris_distribs_OriginalSplit_SMS3G4G'
CsvFilesThresholdsLoc = '/WORKSPACE/Pierre/StefaniaResults/Cancan2022_Paris_Thresholds_OriginalSplit_SMS3G4G_6Thresh.csv'


print(ParquetFilesDistribsLoc)
print(CsvFilesThresholdsLoc)


# load the data
ErrorsDFSP = spark.read.parquet(ErrorsParquetFileLoc)
ErrorsDFSP.printSchema()



#OriginalDataDFSP = spark.read.parquet(SourceParquetFilesLoc).drop('WeeksGroup')
OriginalDataDFSP = spark.read.parquet(SourceParquetFilesLoc)

# generate the sum column on the fly
#OriginalDataDFSP = OriginalDataDFSP.withColumn( "VoicePlusCall", (f.nanvl(OriginalDataDFSP.Voice, f.lit(0)) + f.nanvl(OriginalDataDFSP.Call, f.lit(0))).cast('long') ).fillna(value=0, subset=['VoicePlusCall'])
OriginalDataDFSP.printSchema()









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

#from anr_discret import offlineMLbyPL

import pandas as pd
import numpy as np






import scipy.stats as stats

SigmaSep = 2.32          # how do we select the set of absolute error values to 


def fit_distribution_eric_local(df:pd.DataFrame, metrics:list) -> dict:
    """Fit a Gamma distribution to the given columns the input dataframe.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The input dataframe (usually containing AE values)
    metrics: list
        The list of column names to fit the Gamma distribution

    Returns
    -------
    distrib: dict
        The dictionary containing, for each column name, the Gamma parameters
    """

    distrib = {}

    #compter nombre de cas où on est à np.inf pour vérifier que ça n'arrive pas
    df = df.replace([np.inf, -np.inf], np.nan)

    
    for m in metrics:
        # pour éviter sort : hypothèse distribution gaussienne + estimer à partir de mean et std
        # alpha = 2.32 std pour exclure 99% des donnée
        # alpha = 1.5 std pour exclure 95% données
        x_mean = df[m].mean()
        x_std = df[m].std()
        SeparationThresh = x_mean + SigmaSep * x_std
        x = df[m].loc[df[m] > SeparationThresh]
        
        # tout écart inférieur à seuil aura comme probabilité 1-proba
        proba = len(x.index) / len(df.index)
        
        # donc on stocke seuil et on compare dans la phase de détection à seuil
        # si value < seuil, on retourne 1-proba
        # sinon, on regarde la sf de la loi et on multiplie par proba
        # regarder la manière dont cette loi est respectée
        firsttuple = (SeparationThresh, proba, len(x.index))

        if len(x.index)>=3:
            # regarder options pour supprimer outliers dans la fonction gamma fit
            fittuple = stats.gamma.fit(x)
            # before the code used to translate this tuple used to be the following:

            distrib[m] = firsttuple + (fittuple[0], fittuple[-2], fittuple[-1])
        else:
            distrib[m] = firsttuple + (np.nan, np.nan, np.nan)
    return distrib







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

    res['ALR'] = als.sum(axis=1)
    return res


distributionsDFList = []
totalALR = []





with tqdm(total=len(LocIdsList)*len(AllWeeksGroupsList), desc='Distributions computation') as pbar:
#with tqdm(total=len(LocIdsList)*1, desc='Signature extraction') as pbar:
        
    # run 2 imbricated loops, this hasn't much consequence on the output thanks to the filter / partitioning thing 
    for WGrp in AllWeeksGroupsList:

        TrainWeeksIdsList = TrainWeeksGroupsList[WGrp]
        print("\nWorking on group " + str(WGrp) + " - corresponding weeks are the following :")
        print(TrainWeeksIdsList)
        
        for LocId in LocIdsList:
            #if LocId>20:
            #    break
            # gather all the data corresponding to the training group and the corresponding location
            # start with the location
            ErrorsLocDFSP = ErrorsDFSP.filter(ErrorsDFSP.LocationId==LocId)

            ErrorsDFsList = []
            for wk in TrainWeeksIdsList:
                wkDF = ErrorsLocDFSP.filter(ErrorsLocDFSP.WeekOfYear==wk).toPandas()
                wkDF = wkDF.set_index('MinuteWithinWeek')
                wkDF = wkDF.reindex(range(0,24*7*60), fill_value=0).assign(WeekOfYear=wk, LocationId=LocId, WeeksGroup=WGrp).fillna(0)
                wkDF.reset_index(inplace=True)

                ErrorsDFsList.append( wkDF )

            errors = pd.concat(ErrorsDFsList)


            #distributions = offlineMLbyPL.get_distrib_params(errors, meta=meta, metrics=metrics)
            distribution = pd.DataFrame(np.nan, index=['thresh', 'proba', 'nvalues', 'k','loc','theta'], columns=['WeeksGroup', 'LocationId', *metrics])
            distrib = fit_distribution_eric_local(errors, metrics)
            for m in metrics:
                for k in range(0,6):
                    mLoc = distribution.columns.get_loc(m)
                    distribution.iloc[k,mLoc] = distrib[m][k]

            distribution = distribution.assign(LocationId=LocId, WeeksGroup=WGrp)
            distribution.index.names = ['GammaParam']

            #alrDF = compute_alr_eric_local(AbsErrors, GammaLawDF, metrics)
            alrDF = compute_alr_eric_local(errors, distribution, metrics)
            # maybe this is what makes the platform crash
            alrDF = alrDF.drop(columns=metrics)

            distribution.reset_index(inplace=True)
            distributionsDFList.append(distribution)


            totalALR.append(alrDF)

            #break


            pbar.update(1)
        #break

        
            
distributionsDF = pd.concat(distributionsDFList)
totalALRDF = pd.concat(totalALR)



distributionsDF.info(verbose=True)
totalALRDF.info(verbose=True)


distributionsDF.to_parquet(path=ParquetFilesDistribsLoc, partition_cols=['WeeksGroup'], index=False)




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



#CsvFilesThresholdsLoc = 'exports/' + DatasetPrefix + '_thresholds_0fill2groups_' + LocationPrefix + '.csv'


# apply the offlineML functions to aggregate thresholds for each antenna
#thresholds = totalALRDF.groupby(['WeeksGroup','LocationId']).ALR.agg([level1byPL, level2byPL, level3byPL])
print(totalALRDF.describe())


thresholds = totalALRDF.groupby(['WeeksGroup','LocationId']).ALR.agg([level1byPL6Thresh, level2byPL6Thresh, level3byPL6Thresh, level4byPL6Thresh, level5byPL6Thresh, level6byPL6Thresh])
thresholds.reset_index(inplace=True)




print(thresholds.info(verbose=True))



thresholds.to_csv(CsvFilesThresholdsLoc, index=False)



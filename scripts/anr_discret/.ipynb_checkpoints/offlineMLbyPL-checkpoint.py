""" ************************** IMPORTS ************************ """
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import butter, filtfilt


""" ************************* VARIABLES *********************** """
T = 7                    # sample period: 7 days
sample_rate = 60*24      # sample rate in Hz: number of samples per day
nyq = 0.5 * sample_rate  # Nyquist Frequency in Hz
order = 2                # sin wave can be approx represented as quadratic
SigmaSep = 2.32          # how do we select the set of absolute error values to 



""" ************************* FUNCTIONS *********************** """
def signature_weekly(df:pd.DataFrame, date_col:str) -> pd.DataFrame:
    """Compresses the input dataframe into a week-long dataframe.
    The input dataframe should be at least one week long, have a date-like column and have a 1-minute granularity.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The dataframe to compress
    date_col: str
        The name of the date-like column of the dataframe

    Returns
    -------
    df_weekly: pandas.core.frame.DataFrame
        The compressed dataframe containing the weekly signature for each column
    """

    df = pd.DataFrame(df).fillna(0)
    dates = pd.to_datetime(df[date_col], format='%Y-%m-%d %H:%M')
    df.set_index(dates, inplace=True)
    df_weekly = df.groupby([df.index.dayofweek, df.index.hour, df.index.minute]).median()
    df_weekly['new_index'] = df_weekly.index.map(lambda t: str(t[0]) + ' ' + str(t[1]) + ' ' + str(t[2]))
    df_weekly.set_index('new_index', inplace=True)    
    df_weekly['Date'] = df_weekly.index
    return df_weekly

def signature_filtered(df:pd.DataFrame, cutoff:int, metrics:list) -> pd.DataFrame:
    """Applies a low-pass filter on the input dataframe.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The dataframe containing the signals to filter  
    cutoff: int
        The desired cutoff frequency of the filter (Hz)
    metrics: list
        The list of the column names corresponding to the signals to filter

    Returns
    -------
    filt_df: pandas.core.frame.DataFrame
        The dataframe of filtered signals
    """
    
    # replicate the data structure
    filt_df = pd.DataFrame(df).fillna(0)
    if len(df.index) > (cutoff+1):
        for m in metrics:
            filt_sig = butter_lowpass_filter(df[m].values, cutoff)
            filt_df[m] = filt_sig
    return filt_df
    
def butter_lowpass_filter(sig:np.ndarray, cutoff:int) -> np.ndarray:
    """Computes the parameters of a Butterworth filter design and applies it to the input signal

    Parameters
    ----------
    sig: np.ndarray
        The signal to filter
    cutoff: int
        The desired cutoff frequency of the filter (Hz)

    Returns
    -------
    y: np.ndarray
        The filtered signal
    """

    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, sig)
    return y

def absolute_error(df:pd.DataFrame, meta:list, metrics:list) -> pd.DataFrame:
    """Computes the Absolute Error (AE) with respect to the reference signature over the training dataset.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The dataframe of training data. For each data column it should have a reference column '_ref'
    meta: list
        The list of metadata column names
    metrics: list
        The list of metric column names

    Returns
    -------
    errors: pandas.core.frame.DataFrame
        The dataframe of AE values for each column
    """

    errors = pd.DataFrame(index = df.index, columns = [*meta, *metrics])
    errors[meta] = df[meta]
    for m in metrics:
        errors[m] = abs(df[m] - df[m+'_ref'])
    return errors



def fit_distribution(df:pd.DataFrame, metrics:list) -> dict:
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
        # pour éviter sort : hypothèse distribution gaussienne + estimer 
        # alpha = 2.32 std pour exclure 99% des donnée
        # alpha = 1.5 std pour exclure 95% données
        x_mean = df[m].mean()
        x_std = df[m].std()
        seuil = x_mean + alpha * x_std
        #x = df[m].loc[df[m] <= df[m].quantile(0.85)]
        x = df[m].loc[df[m] >= seuil]
        
        # tout écart inférieur à seuil aura comme probabilité 1-proba
        proba = len(x.index) / len(df.index)
        
        # donc on stocke seuil et on compare dans la phase de détection à seuil
        # si value < seuil, on retourne 1-proba
        # sinon, on regarde la sf de la loi et on multiplie par proba
        # regarder la manière dont cette loi est respectée

        if len(x.index)>=3:
            # regarder options pour supprimer outliers dans la fonction gamma fit
            distrib[m] = stats.gamma.fit(x)
        else:
            distrib[m] = (np.nan, np.nan, np.nan)
    return distrib



def fit_distribution_eric(df:pd.DataFrame, metrics:list) -> dict:
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
        firsttuple = (SeparationThresh, proba, len(x.index),)

        if len(x.index)>=3:
            # regarder options pour supprimer outliers dans la fonction gamma fit
            distrib[m] = firsttuple + stats.gamma.fit(x)
        else:
            distrib[m] = firsttuple + (np.nan, np.nan, np.nan)
    return distrib



def compute_alr(df:pd.DataFrame, distrib:dict, metrics:list) -> pd.DataFrame:
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

    res = pd.DataFrame(df)
    als = pd.DataFrame(index = res.index, columns = metrics)
    for m in metrics:
        arg = distrib[m].iloc[0]
        loc = distrib[m].iloc[1]
        scale = distrib[m].iloc[2]
        if arg:
 
            # peut-être seuiller pour empêcher des valeurs trop faibles (par ex 1x / an) pour éviter valeur 0
            # en gros : prendre en compte le cas où la valeur a atteint la limite de précision du calcul
            # et d'abord : identifier ces situations / regarder la fréquence à laquelle on dépasse
            # certains seuils
            
            #x = 1-stats.gamma.cdf(res[m], arg, loc=loc, scale=scale)
            x = stats.gamma.sf(res[m], arg, loc=loc, scale=scale)
        else:
            #x = 1-stats.gamma.cdf(res[m], loc=loc, scale=scale)
            x = stats.gamma.sf(res[m], loc=loc, scale=scale)
        als[m] = pd.Series(np.log(x), index=res.index)
        res[m] = np.log(x)
    res['ALR'] = als.sum(axis=1)
    return res

def get_distrib_params(df:pd.DataFrame, meta:list, metrics:list) -> pd.DataFrame:
    """Creates a separate dataframe containing the Gamma distribution parameters for the given columns.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The input dataframe (usually containing AE values)
    meta: list
        The metadata column names used to identify the dataframe
    metrics: list
        The list of column names corresponding to the service data

    Returns
    -------
    distribution: pandas.core.frame.DataFrame
        The dataframe with [k, loc, theta] lines containing the parameters for each service data
    """

    distribution = pd.DataFrame(index = ['k','loc','theta'], columns=[*meta,*metrics])
    coords = df[meta].iloc[0]
    for c in meta:
        distribution[c] = coords[c]
    distrib = fit_distribution(df, metrics)
    for m in metrics:
        params = distrib[m]
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        distribution[m].iloc[0] = arg[0]
        distribution[m].iloc[1] = loc
        distribution[m].iloc[2] = scale
    return distribution

def set_threshold(alr:pd.Series, q:float) -> float:
    """Sets an ALR threshold for the input data series.

    Parameters
    ----------
    alr: pandas.core.frame.Series
        The series of ALR values
    q: float
        The ALR quantile for the threshold, corresponds to the detection rate
    
    Returns
    -------
    thresh: float
        The ALR threshold of the series
    """

    res = alr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    thresh = res.quantile(q)
    return thresh

def level1(alr:pd.Series) -> float:
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

    res = alr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    # 1 alert every 22 minutes was the parameter used by 
    #return res.quantile(1-0.9546)
    return res.quantile(1-0.9546)

def level2(alr:pd.Series) -> float:
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

    res = alr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    return res.quantile(1-0.9973)

def level3(alr:pd.Series) -> float:
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
    res = alr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    #return res.quantile(1-0.999937)
    return res.quantile(1-(1/10080))





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

    res = alr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    # 1 alert every 22 minutes was the parameter used by 
    # once per 4 hours -> 1 per 4*60 minutes
    #return res.quantile(1/240)
    return res.quantile(0.00416667)

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

    res = alr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    #return res.quantile(1-0.9973)
    # once per day -> 1 per 24*60 minutes
    #return res.quantile(1/1440)
    return res.quantile(0.0006944)

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
    res = alr.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    #return res.quantile(1-0.999937)
    return res.quantile(0.0000992)

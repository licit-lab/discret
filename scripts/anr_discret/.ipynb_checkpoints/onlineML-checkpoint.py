""" ************************** IMPORTS ************************ """
import numpy as np
import pandas as pd
import scipy.stats as stats


""" ************************* FUNCTIONS *********************** """
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
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    for m in metrics:
        x = df[m].loc[df[m] <= df[m].quantile(0.85)]
        distrib[m] = stats.gamma.fit(x)
    return distrib

def compute_alr(df:pd.DataFrame, distrib:pd.DataFrame, metrics:list):
    """Computes the Anomaly Likelihood Rate (ALR) over the input dataframe.
    The ALR corresponds to the sum of the logs of the p-value for each service data.
    The p-value is obtained for each service data by fitting a Gamma distribution over the dataset.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The input dataframe (usually containing AE values)
    distrib: pandas.core.frame.DataFrame
        The dataframe containing the distribution parameters for each service data
    metrics: list
        The list of column names corresponding to the service data
    """

    df['ALR'] = 0
    als = ['AL_'+m for m in metrics]
    for m in als:
        df[m] = 0
    for m in metrics:
        arg = distrib.loc['k',m]
        loc = distrib.loc['loc',m]
        scale = distrib.loc['theta',m]
        if arg:
            x = 1-stats.gamma.cdf(df[m], arg, loc=loc, scale=scale)
        else:
            x = 1-stats.gamma.cdf(df[m], loc=loc, scale=scale)
        ser = pd.Series(np.log(x), index=df.index)
        df['AL_'+m] = ser
    df['ALR'] = df[als].sum(axis=1)

def replace_inf(df, metrics, thresh):
    """Replace infinite ALR values in the dataframe for processing purposes.

    Parameters
    ----------
    df: pandas.core.frame.DataFrame
        The dataframe containing the ALR values
    metrics: list
        The list of column names corresponding to the service data
    thresh: pandas.core.frame.Series
        The ALR thresholds by anomaly level corresponding to the dataframe

    Returns
    -------
    res: pandas.core.frame.DataFrame
        The dataframe with finite ALR values
    """

    res = df.copy()
    als = ['AL_'+m for m in metrics]
    rep = thresh.level3/3  # Maximal anomaly if at least 3 service ALs are infinite
    res[als] = res[als].replace(-np.inf, rep)
    res['ALR'] = res[als].sum(axis=1)
    return res

def get_threshold(key, thresholds):
    """Get the ALR thresholds by anomaly level corresponding to the dataframe.

    Parameters
    ----------
    key: str
        The identification key of the dataframe
    thresholds: pandas.core.frame.DataFrame
        the dataframe containing all ALR thresholds

    Returns
    -------
    thresh: pandas.core.frame.Series
        The ALR thresholds corresponding to the dataframe
    """

    return thresholds.loc[thresholds.index==key].iloc[0]
    
def set_levels(df, thresh):
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
    df.loc[df.ALR<=thresh.level1, 'Anomaly_Level'] = 1
    df.loc[df.ALR<=thresh.level2, 'Anomaly_Level'] = 2
    df.loc[df.ALR<=thresh.level3, 'Anomaly_Level'] = 3
    return df

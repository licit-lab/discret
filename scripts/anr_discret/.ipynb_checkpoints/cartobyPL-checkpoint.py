""" ************************** IMPORTS ************************ """
import folium
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from folium.plugins import HeatMapWithTime
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from anr_discret import onlineML


""" ************************* FUNCTIONS *********************** """
def concat_df(signals:dict, month:int, days:list) -> pd.DataFrame:
    """Concatenate all data of the geographical area for the given date.

    Parameters
    ----------
    signals: dict
        The dictionary containing local pandas Dataframes
    month: int
        The month of the date
    days: list
        The day(s) of the date

    Returns
    -------
    full_df: pd.DataFrame
        The concatenated data at the given date
    """

    for k,df in signals.items():
        signals[k] = df.loc[(df.index.month==month) & (df.index.day.isin(days))]
    full_df = pd.concat([df for df in signals.values()])
    return full_df.sort_index()

def concat_scales(signals:dict, thresholds:pd.DataFrame, month:int, days:list) -> pd.DataFrame:
    """Convert the ALR values into anomaly intensity indicators and concatenate local data, at a given date.

    Parameters
    ----------
    signals: dict
        The dictionary containing local data
    thresholds: pd.DataFrame
        The dataframe containing the ALR thresholds for each dataframe
    month: int
        The month of the date
    days: list
        The day(s) of the date

    Returns
    -------
    full_df: pd.DataFrame
        The concatenated data at the given date with scaled ALR
    """

    for k,df in signals.items():
        thresh = onlineML.get_threshold(k, thresholds)
        df.ALR = df.ALR.replace(-np.inf, thresh)
        df['scale'] = abs(df.ALR)/abs(thresh)
        df['scale'] = df['scale'].where(df['scale'] <= 1, 1)
        df['scale'] = df['scale'].where(df['scale'] >= 0.01, 0.01)
    return concat_df(signals, month, days)

def add_enb(enb:dict) -> folium.FeatureGroup:
    """Add eNode-B locations on the map. The input data must have LAT and LON fields.

    Parameters
    ----------
    enb: dict
        The dictionary containing eNode-B locations to add
    
    Returns
    -------
    fg: folium.FeatureGroup
        The Folium layer of eNode-B locations
    """

    fg = folium.FeatureGroup(name='eNode-B')
    for k,df in enb.items():
        if len(df)>0:
            folium.features.RegularPolygonMarker(
                location = [df.LAT.iloc[0], df.LON.iloc[0]],
                popup = k,
                number_of_sides = 3,
                radius = 10,
                color = 'black',
                fill_color = 'black',
                opacity = 1,
                rotation = 30
            ).add_to(fg)
    return fg
    
def create_heatdata(full_df:pd.DataFrame) -> list:
    """Creates HeatMap-compliant data from the input dataframe. The input dataframe must have LAT, LON and scale fields.

    Parameters
    ----------
    full_df: pd.DataFrame
        The input data to reshape
    
    Returns
    -------
    heat_data: list
        A list of ['LAT','LON','scale'] fields for the HeatMap
    """

    heat_data = []
    agg = full_df.groupby(full_df.index)
    for k in agg.groups.keys():
        df = agg.get_group(k)
        data = df[['LAT','LON','scale']].values.tolist()
        heat_data.append(data)
    return heat_data

def create_heatmap(full_df:pd.DataFrame) -> HeatMapWithTime:
    """Creates a continuous HeatMap with time of the ALR evolution in a geographical area.
    The input dataframe must have LAT, LON and scale fields.

    Parameters
    ----------
    full_df: pd.DataFrame
        The input dataframe containing scaled ALR values with locations
    
    Returns
    -------
    hm: folium.plugins.HeatMapWithTime
        the Folium HeatMap layer
    """

    heat_data = create_heatdata(full_df)
    index = [t.ctime() for t in sorted(full_df.index.unique())]
    hm = HeatMapWithTime(
        data = heat_data,
        index = index,
        name = 'Anomaly spread', 
        radius = 60,
        gradient = {0.1:'blue', 0.5:'lime', 0.9:'orange', 1:'red'},
        min_opacity = 0.2, 
        max_opacity = 0.9,
        index_steps = 1,
        min_speed = 1,
        max_speed = 30,
        speed_step = 1
    )
    return hm

def create_map(location:tuple, data:pd.DataFrame, enb=None) -> folium.Map:
    """Creates a Folium map of the geographical area.

    Parameters
    ----------
    location (x,y): tuple
        The latitude and longitude (WGS-84) of the center of the map
    data: pd.DataFrame
        The input data to use for the creation of the HeatMap layer
    enb: dict, optional
        The dictionary of eNode-B locations to add to the map

    Returns
    -------
    map: folium.Map
        The map with HeatMap (and eNode-B) layer(s)
    """

    ly_map = folium.Map(location, zoom_start=13)
    if enb:
        add_enb(enb).add_to(ly_map)
    create_heatmap(data).add_to(ly_map)
    folium.TileLayer('stamentoner').add_to(ly_map)
    folium.LayerControl().add_to(ly_map)
    return ly_map

def display_local(rlt:pd.DataFrame, err:pd.DataFrame, thresh:pd.Series, metrics:list, hstart:datetime, hstop:datetime, dates:list):
    """Creates a complex figure with the real-time evolution of metrics, the ALR avolution and the detection delay of the anomaly detection at a given node.

    Parameters
    ----------
    rlt: pd.DataFrame
        The dataframe of real-time signals at a given node
    err: pd.DataFrame
        The dataframe of AE values at the same node
    thresh: float
        The ALR threshold corresponding to the input AE dataframe
    metrics: list
        The list of service columns to analyse
    hstart: datetime.datetime
        The date to start the analysis
    hstop: datetime.datetime
        The date to stop the analysis
    anom_start: datetime.datetime
        The date of occurrence of the anomaly (ground-truth)
    anom_detect: datetime.datetime
        The date of the first anomaly detection by the system
    """

    fig = plt.figure(figsize=(16,8))
    gs = GridSpec(nrows=len(metrics), ncols=2)

    rlt = rlt.sort_index()
    err = err.sort_index()
    df = rlt.loc[(rlt.index>=hstart) & (rlt.index<hstop)]
    idx = df.index
    hform = matplotlib.dates.DateFormatter('%H:%M')
    for i,m in enumerate(metrics):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(idx, df[m].values, lw=1, color='blue')
        ax.plot(idx, df[m+'_ref'].values, lw=2, color='lightblue')
        ax.scatter(df.loc[err.Anomaly_Level==1].index, df[m].loc[err.Anomaly_Level==1], color='yellow')
        ax.scatter(df.loc[err.Anomaly_Level==2].index, df[m].loc[err.Anomaly_Level==2], color='orange')
        ax.scatter(df.loc[err.Anomaly_Level==3].index, df[m].loc[err.Anomaly_Level==3], color='red')
        ax.axvline(x=dates[0], color='orange', ls='--', lw=1)
        ax.axvline(x=dates[1], color='red', ls='--', lw=1)
        ax.axvline(x=dates[2], color='darkred', ls='--', lw=1)
        
        legend_lines = [Line2D([0], [0], color='blue', ls='-', lw=1),
                        Line2D([0], [0], color='lightblue', ls='-', lw=2),
                        Line2D([0], [0], color='orange', ls='--', lw=1),
                        Line2D([0], [0], color='red', ls='--', lw=1),
                        Line2D([0], [0], color='darkred', ls='--', lw=1)]
        
        ax.legend(legend_lines, [m, m+' signature','Pre-alert','Alert','Max alert'], loc='upper left')
        ax.set(xlabel='Time', ylabel='Volume')
        ax.xaxis.set_major_formatter(hform)
        ax.label_outer()

    ax2 = fig.add_subplot(gs[:, 1])
    df = err.loc[(err.index>=hstart) & (err.index<hstop)]
    idx = df.index
    for m in metrics:
        ax2.plot(idx, df['AL_'+m].values, ls=':', lw=1)
    ax2.plot(idx, df.ALR.values, color='blue', ls='-', lw=1)
    ax2.axhline(y=thresh.level1, color='orange', ls='--', lw=2)
    ax2.axhline(y=thresh.level2, color='red', ls='--', lw=2)
    ax2.axhline(y=thresh.level3, color='darkred', ls='--', lw=2)

    legend_lines = [Line2D([0], [0], color='blue', ls='-', lw=1),
                    Line2D([0], [0], color='orange', ls='--', lw=2),
                    Line2D([0], [0], color='red', ls='--', lw=2),
                    Line2D([0], [0], color='darkred', ls='--', lw=2)]
    ax2.legend(legend_lines, ['ALR','PA threshold','A threshold','MA threshold'], loc='lower left')
    ax2.set(xlabel='Time', ylabel='Likelihood rate')
    ax2.xaxis.set_major_formatter(hform)
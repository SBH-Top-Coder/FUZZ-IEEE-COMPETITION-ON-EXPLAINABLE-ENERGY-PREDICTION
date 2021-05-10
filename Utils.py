import numpy as np
import pandas as pd
from tqdm import tqdm

def load (path) :
  df = pd.read_csv(path)
  return(df)

def impute_na(df, variable):
    # extract the random sample to fill the na
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0,replace=True)
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable] = random_sample
def prepare_add(cons,add)  :
  add = add[['meter_id','dwelling_type','num_bedrooms']]
  df = cons['meter_id']
  df = pd.DataFrame(df)
  df['dwelling_type'] = '_'
  df['num_bedrooms'] = -1
  for x in df['meter_id'].values :
    try :
      dwelling_type = add[add['meter_id']==x]['dwelling_type'].values[0]
      num_bedrooms = add[add['meter_id']==x]['num_bedrooms'].values[0]
      df.at[df[df['meter_id']==x].index[0] ,'dwelling_type'] = dwelling_type
      df.at[df[df['meter_id']==x].index[0] ,'num_bedrooms'] = num_bedrooms

    except :
      pass
  df['dwelling_type'] = df['dwelling_type'].fillna('_') # lhna héthi lézma fl num_bedroom ki tabda nan valeur li 5arajnéha
  return(df)

def feature_worktime(row):
    if row["hour"] > 7 and row["hour"] <= 17:
        return "Worktime"
    else :
      return "NonWorkTime"
def feature_peak(row):
    if row["hour"] >= 17 and row["hour"] <= 20:
        return "Peak"
    else :
      return "NonPeak"
def feature_weekend(row):
    if row["day"] == 5 or row["day"] == 6 or row["day"] == 7:
        return "Weekend"
    else :
      return "NonWeekend"
def feature_vis1(row) :
  if row["month"] == 11 or row["month"] == 12 or row["month"] == 1 or row["month"] == 2 :
    return "bard"
  else :
    return "3adi"
def submit(columns,cons,pred,explanations) :
    df1 = pd.DataFrame([], columns=columns)
    for i in range(3248):
        meter_id = cons.iloc[i].values[0]
        L = []
        conso = pred[17520 * i:17520 * (i + 1)]
        Janv_pred  = conso[0:1488].sum()
        L.append(Janv_pred)
        JanEXP = explanations[0+i*12]
        Fev_pred = conso[1488: 2832].sum()
        L.append(Fev_pred)
        FebEXP = explanations[1+i*12]
        Mars_pred = conso[2832: 4320].sum()
        L.append(Mars_pred)
        MarEXP = explanations[2+i*12]
        Apr_pred = conso[4320: 5760].sum()
        L.append(Apr_pred)
        AprEXP = explanations[3+i*12]
        Mai_pred = conso[5760: 7248].sum()
        L.append(Mai_pred)
        MayEXP = explanations[4+i*12]
        Jui_pred = conso[7248: 8688].sum()
        L.append(Jui_pred)
        JunEXP = explanations[5+i*12]
        Juil_pred = conso[8688: 10176].sum()
        L.append(Juil_pred)
        JulEXP = explanations[6+i*12]
        Aug_pred = conso[10176: 11664].sum()
        L.append(Aug_pred)
        AugEXP = explanations[7+i*12]
        Sep_pred = conso[11664: 13104].sum()
        L.append(Sep_pred)
        SepEXP = explanations[8+i*12]
        Oct_pred = conso[13104: 14592].sum()
        L.append(Oct_pred)
        OctEXP = explanations[9+i*12]
        Nov_pred = conso[14592:16032].sum()
        L.append(Nov_pred)
        NovEXP = explanations[10+i*12]
        Dec_pred = conso[16032: 17520].sum()
        L.append(Dec_pred)
        DecEXP = explanations[11+i*12]
        Months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                  'October', 'November', 'December']
        Annual_pred = conso.sum()
        AnnualEXP = 'There will be a big consumption on ' + Months[L.index(max(L))] + ' and a low consumption on ' + Months[L.index(min(L))]
        df_tmp = pd.DataFrame([(meter_id, Annual_pred,AnnualEXP ,  Janv_pred,JanEXP, Fev_pred,FebEXP , Mars_pred,
    MarEXP ,Apr_pred,AprEXP, Mai_pred,MayEXP, Jui_pred,JunEXP,Juil_pred,JulEXP, Aug_pred,AugEXP, Sep_pred,
    SepEXP, Oct_pred,OctEXP, Nov_pred,NovEXP, Dec_pred,DecEXP)], columns=columns)

        df1 = df1.append(df_tmp)
    return (df1)

def month_to_text(rule) :
  L = rule.split()
  index = L.index('month')
  Months =  ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
              'October','November', 'December']

  if (index == 0 )  : # on est donc dans le cas d'une régle ss la forme ..<...
    if (L[1] == '<=') :
      L = Months [:int(float(L[2]))]
      ch = ' month is in '
      ch1 = ''
      for x in L :
        ch1 = ch1 + x + ' , '
      ch = ch + ch1
    elif (L[1] == '<'):
      L = Months [:int(float(L[2]))-1]
      ch = ' month is in '
      ch1 = ''
      for x in L :
        ch1 = ch1 + x + ' , '
      ch = ch + ch1
    elif (L[1]=='>=')  :
      L = Months [int(float(L[2]))-1:]
      ch = ' month is in '
      ch1 = ''
      for x in L :
        ch1 = ch1 + x + ' , '
      ch = ch + ch1
    else :
      L = Months [int(float(L[2])):]
      ch = ' month is in '
      ch1 = ''
      for x in L :
        ch1 = ch1 + x + ' , '
      ch = ch + ch1
  else :
      ch = 'month is between ' + Months [int(float(L[0]))-1] + ' and '  +  Months [int(float(L[4]))-1] + ' , '

  return ch[:len(ch)-2]

def season_to_text(rule) :
  rule = rule.split()
  L = rule.copy()
  index = rule.index('season')
  season = ['winter','spring','summer','autumn']
  if (index == 0 )  : # on est donc dans le cas d'une régle ss la forme ..<...
    if (L[1] == '<=') :
      L = season [:int(float(L[2]))]
      ch = ' season is in '
      ch1 = ''
      for x in L :
        ch1 = ch1 + x + ' , '
      ch = ch + ch1
    elif (L[1] == '<'):
      L = season [:int(float(L[2]))-1]
      ch = ' season is in '
      ch1 = ''
      for x in L :
        ch1 = ch1 + x + ' , '
      ch = ch + ch1
    elif (L[1]=='>=')  :
      L = season [int(float(L[2])):]
      ch = ' season is in '
      ch1 = ''
      for x in L :
        ch1 = ch1 + x + ' , '
      ch = ch + ch1
    else :
      L = season [int(float(L[2]))+1:]
      ch = ' season is in '
      ch1 = ''
      for x in L :
        ch1 = ch1 + x + ' , '
      ch = ch + ch1
  else :
    ch = 'the season is between ' + season[int(float(rule[0]))-1] + ' and ' + season[int(float(rule[4]))-1]

  return(ch[:len(ch)-2])


def weather_cluster_to_text(rule):
    L = rule.split()
    L1 = ['increasing_Temperature', 'Hot_Temperature', 'Decreasing_Temperature']
    ch = ' we are in the ' + L1[int(float(L[-1]))-1]
    return (ch)

def dwelling_type_to_text(rule) :
  L = rule.split()
  L1 = ['semi_detached_house','detached_house','terraced_house','bungalow','flat','_']
  ch = ' dwelling_type is  ' + L1[int(float(L[-1]))]
  return(ch)


def num_bedrooms_to_text(rule):
    L = rule.split()
    ch = ' number of bedrooms is equal to  ' + str(int(float(L[-1])))
    return (ch)


def denya_berda_to_text(rule):
    L = rule.split()
    if (int(float(L[-1]) == 1)):
        ch = ' the weather is cold '
    else:
        ch = ' the weather is not cold '

    return (ch)

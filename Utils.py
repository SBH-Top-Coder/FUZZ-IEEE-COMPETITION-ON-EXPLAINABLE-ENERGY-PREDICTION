import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import copy

def load (path) : # fonction nloadiw biha df
  df = pd.read_csv(path)
  return df


def random_sampling(df, variable): # ta93ad fonction héthi khatér  bch nésta3mlouha bch nfiliw nanét ta3 weather
    # 5atérhom chwaya ...
    # extract the random sample to fill the na
    random_sample = df[variable].dropna().sample(df[variable].isnull().sum(), random_state=0,replace=True)
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = df[df[variable].isnull()].index
    df.loc[df[variable].isnull(), variable] = random_sample

def impute_na(cons , janv_moy , fev_moy , mars_moy , avril_moy , may_moy , juin_moy , juil_moy , aout_moy, sept_moy ,oct_moy, nov_moy,dec_moy  ) :
    # fonction li éntouma talbinha ... ml janv_moy hata ll dec_moy hékom des compteurs moyen représentant chaque mois
    # l'explication ta3 san3anhom tal9aha fl main
    for l, meter in cons.iterrows() :
        meter = pd.DataFrame(meter[1:])
        meter.columns = ['consommation']
        if ((meter.isnull().sum() / 17520).values[0] > 0.6):
        # kénha a9al mn 60/100
            s = -1  # contient le numéro de la ligne courante
            for index, row in meter.iterrows(): # éjbéd bl ligna bl ligne
                s = s + 1
                x = row['consommation']
                if (math.isnan(x)): # kén l9it valeur nan
                    if (pd.to_datetime(index).strftime("%A") in ['Sunday', 'Saturday']): # kén hna ahna wost week end
                        i = 0  # 9adéh mn mara bch nwa5ar
                        a = copy.deepcopy(s)  # compteur li bch yémchi yéjbéd lignét ltéli wili 9odém ...
                        b = copy.deepcopy(s)
                        while (math.isnan(meter.iloc[s]['consommation']) and (i < 3)):  # nchouf les valeurs ta3
                            # sé3a 9odém w sé3a téli
                            i = i + 1
                            a = a - 1
                            b = b + 1
                            try:
                                v1 = meter.iloc[a]['consommation']
                                v2 = meter.iloc[b]['consommation']
                                if ((math.isnan(v1) == False)):
                                    cons.at[l,index] = v1
                                    break
                                elif ((math.isnan(v2) == False)):
                                    cons.at[l, index] = v2
                                    break
                            except:
                                pass

                        i = 0
                        a = copy.deepcopy(s) - 336  # nwa5ro b jom3a taw (168 sé3a * 2 = 336 nos sé3a fl jom3a )
                        b = copy.deepcopy(s) + 336
                        while (math.isnan(meter.iloc[s]['consommation']) and (i < 3)):
                        # narj3o joma ltéli w nchoufo famchi des valeurs mahomch nans fi nafs sé3a mn jom3a ltéli w kodém
                        # w sé3a kodém w sé3a ltéli fi jom3a ltéli w jom3a 9odém
                            i = i + 1
                            a = a - 1
                            b = b + 1
                            try:
                                v1 = meter.iloc[a]['consommation']
                                v2 = meter.iloc[b]['consommation']
                                if ((math.isnan(v1) == False)):
                                    cons.at[l, index] = v1
                                    break
                                elif ((math.isnan(v2) == False)):
                                    cons.at[l, index] = v2
                                    break
                            except:
                                pass

                        i = 0
                        a = copy.deepcopy(s) - 672  # kif kif kima 336 ... ama hna jom3tin ltéli w jom3tin kodém
                        b = copy.deepcopy(s) + 672
                        while (math.isnan(meter.iloc[s]['consommation']) and (i < 3)):
                            i = i + 1
                            a = a - 1
                            b = b + 1
                            try:
                                v1 = meter.iloc[a]['consommation']
                                v2 = meter.iloc[b]['consommation']
                                if ((math.isnan(v1) == False)):
                                    cons.at[l, index] = v1
                                    break
                                elif ((math.isnan(v2) == False)):
                                    cons.at[l, index] = v2
                                    break
                            except:
                                pass

                        i = 0
                        a = copy.deepcopy(s) - 1008  # kif kif hna 3 jmo3 ltéli w 3 kodém
                        b = copy.deepcopy(s) + 1008
                        while (math.isnan(meter.iloc[s]['consommation']) and (i < 3)):  # sé3a téli ...
                            i = i + 1
                            a = a - 1
                            b = b + 1
                            try:
                                v1 = meter.iloc[a]['consommation']
                                v2 = meter.iloc[b]['consommation']
                                if ((math.isnan(v1) == False)):
                                    cons.at[l, index] = v1
                                    break
                                elif ((math.isnan(v2) == False)):
                                    cons.at[l, index] = v2
                                    break
                            except:
                                pass
                    else: # kén manéch fi week end nchédo nchoufo a9rab nhar ml layém li mahomch week end famchi valeur
                        # m3obia
                        i = 0  #
                        a = copy.deepcopy(s)
                        b = copy.deepcopy(s)
                        while (math.isnan(meter.iloc[s]['consommation']) and (i < 15)):  # sé3a téli ...
                            i = i + 1
                            a = a - 48 * i
                            b = b + 48 * i
                            try:
                                v1 = meter.iloc[a]['consommation']
                                v2 = meter.iloc[b]['consommation']
                                index_a = meter.index[a]
                                index_b = meter.index[b]
                                if ((math.isnan(v1) == False) and (
                                        pd.to_datetime(index_a).strftime("%A") not in ['Sunday', 'Saturday'])):
                                    cons.at[l, index] = v1
                                    break
                                elif ((math.isnan(v2) == False) and (
                                        pd.to_datetime(index_a).strftime("%A") not in ['Sunday', 'Saturday'])):
                                    cons.at[l, index] = v2
                                    break
                            except:
                                pass
        else:
            # kén pourcentage de nans akbar mn 60 /100 : nchédo bl chhar bl chhar né7sbo valeur moyenne ta3 les valeurs
            # existants v1 w né7sbo valeur moyenne de ta3 compteur moyén v2 w né7sbo coef d = v1 - v2
            # w ba3d fkol compteur 3ando des valeurs né9sin fi chhar mo3ayan w nchédo né54ou valeur ta3 compteur moyen fi
            # nafs blasa w na7iw méno d

            # janvier
            d = meter.loc['2017-01-01 00:00:00':'2017-01-31 23:30:00', :]['consommation'].mean() - janv_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-01-01 00:00:00':'2017-01-31 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = janv_moy.iloc[s]['consommation'] + d
            # février
            d = meter.loc['2017-02-01 00:00:00':'2017-02-28 23:30:00', :]['consommation'].mean() - fev_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-02-01 00:00:00':'2017-02-28 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = fev_moy.iloc[s][
                                                                                                            'consommation'] + d
            # mars
            d = meter.loc['2017-03-01 00:00:00':'2017-03-31 23:30:00', :]['consommation'].mean() - mars_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-03-01 00:00:00':'2017-03-31 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = mars_moy.iloc[s]['consommation'] + d
            # avril
            d = meter.loc['2017-04-01 00:00:00':'2017-04-30 23:30:00', :]['consommation'].mean() - avril_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-04-01 00:00:00':'2017-04-30 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = avril_moy.iloc[s]['consommation'] + d
            # may
            d = meter.loc['2017-05-01 00:00:00':'2017-05-31 23:30:00', :]['consommation'].mean() - may_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-05-01 00:00:00':'2017-05-31 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = may_moy.iloc[s]['consommation'] + d
            # juin
            d = meter.loc['2017-06-01 00:00:00':'2017-06-30 23:30:00', :]['consommation'].mean() - juin_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-06-01 00:00:00':'2017-06-30 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = juin_moy.iloc[s]['consommation'] + d
            # juillet
            d = meter.loc['2017-07-01 00:00:00':'2017-07-31 23:30:00', :]['consommation'].mean() - juil_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-07-01 00:00:00':'2017-07-31 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = juil_moy.iloc[s]['consommation'] + d
            # aout
            d = meter.loc['2017-08-01 00:00:00':'2017-08-31 23:30:00', :]['consommation'].mean() - aout_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-08-01 00:00:00':'2017-08-31 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = aout_moy.iloc[s]['consommation'] + d
            # septembre
            d = meter.loc['2017-09-01 00:00:00':'2017-09-30 23:30:00', :]['consommation'].mean() - sept_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-09-01 00:00:00':'2017-09-30 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = sept_moy.iloc[s]['consommation'] + d
            # octobre
            d = meter.loc['2017-10-01 00:00:00':'2017-10-31 23:30:00', :]['consommation'].mean() - oct_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-10-01 00:00:00':'2017-10-31 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = oct_moy.iloc[s]['consommation'] + d
            # novembre
            d = meter.loc['2017-11-01 00:00:00':'2017-11-31 23:30:00', :]['consommation'].mean() - nov_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-11-01 00:00:00':'2017-11-31 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = nov_moy.iloc[s]['consommation'] + d
                    # décembre
            d = meter.loc['2017-12-01 00:00:00':'2017-12-31 23:30:00', :]['consommation'].mean() - dec_moy[
                'consommation'].mean()
            if (math.isnan(d) == True):
                d = 0
            s = - 1
            for index, row in meter.loc['2017-12-01 00:00:00':'2017-12-31 23:30:00', :].iterrows():
                s = s + 1
                if (math.isnan(row['consommation'])):
                    cons.at[l, index] = dec_moy.iloc[s]['consommation'] + d
        #
        # if ((cons.isnull().sum().values[0]) != 0 ) : # kén fo4lét valeur né9sa (probabilité ne dépasse pas 1/100 ) ==> random sampling
        #     random_sample = meter['consommation'].dropna().sample(meter['consommation'].isnull().sum(), random_state=0,replace=True)
        #     # pandas needs to have the same index in order to merge datasets
        #     random_sample.index = meter[meter['consommation'].isnull()].index
        #     meter.loc[meter['consommation'].isnull(), 'consommation'] = random_sample


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num

def prepare_add_wissem (cons,add) :

  L = [cons.columns[(1+i*48)].split()[0] for i in range(365)]
  add = add[['meter_id','dwelling_type','num_bedrooms']]
  nan = add.dwelling_type[4]
  L0 = [x for x in cons.meter_id.values if x not in add.meter_id.values ]
  L1 = [nan for i in range (1105)]
  L2 = [nan for i in range (1105)]
  df = pd.DataFrame({'meter_id':L0 , 'dwelling_type':L1,'num_bedrooms':L2})
  add = pd.concat([add,df],axis=0).reset_index().drop('index',axis=1)
  add['dwelling_type'] = add['dwelling_type'].map({'_':-1 ,'semi_detached_house':1,'detached_house':2,
                                              'terraced_house':3,'bungalow':4,'flat':5 })
  return(add)


def fill_add_kmeans(add, cons):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=130, random_state=0).fit(cons.drop('meter_id', axis=1)) # n_cluster =3248/nb de compteurs par cluster
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = cons.drop('meter_id', axis=1).index.values
    cluster_map['cluster'] = kmeans.labels_
    import math
    for index, row in add.iterrows():
        if (math.isnan(row['dwelling_type'])):
            k = kmeans.predict(
                cons[cons['meter_id'] == row['meter_id']].transpose()[1:].values.reshape(1,
                                                                                                                      -1))[
                0]
            meters = cluster_map[cluster_map['cluster'] == k]['data_index'].values
            L = [add[add['meter_id'] == cons.iloc[meters[i]]['meter_id']]['dwelling_type'].values[0] for i in range(5)]
            add.at[index, 'dwelling_type'] = 2 if (str(most_frequent(L)) == 'nan') else most_frequent(L)

        if (math.isnan(row['num_bedrooms'])):
            k = kmeans.predict(
                cons[cons['meter_id'] == row['meter_id']].transpose()[1:].values.reshape(1,
                                                                                                                      -1))[
                0]
            meters = cluster_map[cluster_map['cluster'] == k]['data_index'].values
            L = [add[add['meter_id'] == cons.iloc[meters[i]]['meter_id']]['num_bedrooms'].values[0] for i in range(5)]
            add.at[index, 'num_bedrooms'] = 2 if (str(most_frequent(L)) == 'nan') else most_frequent(L)
    return (add)


def fill_add_distance(add, cons):
    import math
    from tqdm import tqdm
    for index, row in tqdm(add.iterrows()):
        if (math.isnan(row['dwelling_type'])):
            x = cons[cons['meter_id'] == row['meter_id']].transpose()[1:].values
            dist = [(np.linalg.norm(x - (cons[cons['meter_id'] == y].transpose()[1:].values)), y) for y in
                    cons['meter_id'].values if y != row['meter_id']]
            meters = [pair[1] for pair in sorted(dist, key=lambda x: x[0])[:25]] # 25 a9rab 25 compteurs
            L = [add[add['meter_id'] == meters[i]]['dwelling_type'].values[0] for i in range(5)]
            add.at[index, 'dwelling_type'] = 2 if (str(most_frequent(L)) == 'nan') else most_frequent(L)

        if (math.isnan(row['num_bedrooms'])):
            x = cons[cons['meter_id'] == row['meter_id']].transpose()[1:].values
            dist = [(np.linalg.norm(x - (cons[cons['meter_id'] == y].transpose()[1:].values)), y) for y in
                    cons['meter_id'].values if y != row['meter_id']]
            meters = [pair[1] for pair in sorted(dist, key=lambda x: x[0])[:25]] # a9rab 25 compteurs
            L = [add[add['meter_id'] == meters[i]]['num_bedrooms'].values[0] for i in range(5)]
            add.at[index, 'num_bedrooms'] = 2 if (str(most_frequent(L)) == 'nan') else most_frequent(L)
    return (add)

#
# def prepare_add_semi(cons,add)  : # fonction n5arjo biha li nést7a9ouh ml add info
#   add = add[['meter_id','dwelling_type','num_bedrooms']]
#   df = cons['meter_id']
#   df = pd.DataFrame(df)
#   df['dwelling_type'] = '_'
#   df['num_bedrooms'] = -1
#   for x in df['meter_id'].values :
#     try :
#       dwelling_type = add[add['meter_id']==x]['dwelling_type'].values[0]
#       num_bedrooms = add[add['meter_id']==x]['num_bedrooms'].values[0]
#       df.at[df[df['meter_id']==x].index[0] ,'dwelling_type'] = dwelling_type
#       df.at[df[df['meter_id']==x].index[0] ,'num_bedrooms'] = num_bedrooms
#
#     except :
#       pass
#   df['dwelling_type'] = df['dwelling_type'].fillna('_') # lhna héthi lézma fl num_bedroom ki tabda nan valeur li 5arajnéha
#   return(df)

def feature_worktime(row): # fonction tzid feature ta3 work time ml 7 ll 5 ta3 la3chia
    if row["hour"] > 7 and row["hour"] <= 17:
        return "Worktime"
    else :
      return "NonWorkTime"
def feature_peak(row): # fonction tzid feature ta3 peak .. ml 5 ll 8 ta3 lil
    if row["hour"] >= 17 and row["hour"] <= 20:
        return "Peak"
    else :
      return "NonPeak"
def feature_weekend(row): # fonction tkolik inti tw fil week end wla

    if row["day"] == 5 or row["day"] == 6 or row["day"] == 7:
        return "Weekend"
    else :
      return "NonWeekend"
def feature_vis1(row) : # feature ll dénya bérda wla .. mn novembre l janvier
  if row["month"] == 11 or row["month"] == 12 or row["month"] == 1 or row["month"] == 2 :
    return "bard"
  else :
    return "3adi"

def submit_day (cons,pred,explanations_day) :
    for i in range(1): # nombre de compteurs
        meter_id = cons.iloc[i].values[0]
        conso = pred[17520 * i:17520 * (i + 1)]
        predictions = [conso[48*j:48*(j+1)].sum() for j in range (365)] # prediction pr un seul compteur
        explications = explanations_day[365*i:365*(i+1)]
        ch = 'Let us explain day per day '
        for j in range (365) :
            ch = ch + 'Prediction of the consumption of ' + cons.columns[(1+j*48)].split()[0] + str(predictions[j])
            ch = ch + explications[j]
            textday = open('submit_day.txt', 'a')
            textday.write(ch)
            textday.close()
def submit_week (cons,pred,explanations_week) :
    for i in range(1): # nombre de compteurs
        meter_id = cons.iloc[i].values[0]
        conso = pred[17520 * i:17520 * (i + 1)]
        predictions = [conso[336*j:336*(j+1)].sum() for j in range (52)] # prediction pr un seul compteur
        explications = explanations_week[52*i:52*(i+1)]
        ch = 'Let us explain day per day '
        for j in range (52) :
            ch = ch + 'Prediction of the consumption of ' + cons.columns[(1+j*48)].split()[0] + str(predictions[j])
            ch = ch + explications[j]
            textweek = open('submit_week.txt', 'a')
            textweek.write(ch)
            textweek.close()



def submit_month(columns,cons,pred,explanations,weather)  : # fonction bch t7a4rélna submit file .. té5o prédiction
    # !!!!!!!!!!!!!!!!!! raja3 weather ki traja3 annual EXP nchalah :!!!!!!!!!!! ######
    # w explications w tlasa9hom fi fichier submit
    # df1 = pd.DataFrame([], columns=columns)
    P = []
    for i in range(1):
        meter_id = cons.iloc[i].values[0]
        conso = pred[48 * i:48 * (i + 1)]

        A = []
        weather_co = pd.DataFrame(weather[weather['meter_id'] == meter_id])
        A.append(weather_co.loc[:, '2017-01-01 00:00:00':'2017-01-31 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-02-01 00:00:00':'2017-02-28 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-03-01 00:00:00':'2017-03-31 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-04-01 00:00:00':'2017-04-30 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-05-01 00:00:00':'2017-05-31 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-06-01 00:00:00':'2017-06-30 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-07-01 00:00:00':'2017-07-31 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-08-01 00:00:00':'2017-08-31 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-09-01 00:00:00':'2017-09-30 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-10-01 00:00:00':'2017-10-30 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-11-01 00:00:00':'2017-11-30 00:00:00'].values.mean())
        A.append(weather_co.loc[:, '2017-12-01 00:00:00':'2017-12-31 00:00:00'].values.mean())
        A = np.array(A)
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
        pourcentage = ((A - A.mean())) / A.mean()
        hot_months = (pourcentage > 0).astype('int64')  # en 0/1
        pourcentage = pourcentage * hot_months  # les pourcentage >0
        index = [list(pourcentage).index(x) for x in pourcentage if x != 0]
        hot_months = [Months[i] for i in index]
        pourcentage_final = [x for x in pourcentage if x != 0]

        AnnualEXP = 'There will be a big consumption on ' + Months[L.index(max(L))] + ' and a low consumption on ' + Months[L.index(min(L))] + ' also on ' + str(hot_months)  +'there a big temperature with respectively respectively a percentage overrun of ' + str(pourcentage_final)


        df_tmp = pd.DataFrame([(meter_id, Annual_pred,AnnualEXP ,  Janv_pred,JanEXP, Fev_pred,FebEXP , Mars_pred,
    MarEXP ,Apr_pred,AprEXP, Mai_pred,MayEXP, Jui_pred,JunEXP,Juil_pred,JulEXP, Aug_pred,AugEXP, Sep_pred,
    SepEXP, Oct_pred,OctEXP, Nov_pred,NovEXP, Dec_pred,DecEXP)], columns=columns)

        df1 = df1.append(df_tmp)
    return (df1)


# bloc ta3 fonctionét héthom yé5o rule w yassoci lilha rule maktouba ka texte
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
    elif L[1]=='>='   :
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

def day_to_int(cons) :
  d = {}
  for i in range (365) :
    d[pd.to_datetime(cons.columns[48*i+1]).timetuple().tm_yday ] = cons.columns[1+48*i].split()[0]
  return (d)

def dayofyear_to_text(rule,d) :
  rule = rule.split()
  L = rule.copy()
  index = rule.index('dayofyear')
  ch = ''
  if (index == 0 )  : # on est donc dans le cas d'une régle ss la forme ..<...
    if (L[1] == '<=') or (L[1] == '<') :
      ch = ch + ' the day is before ' + d[int((float(L[2])))]
    else   :
      ch = ch + ' the day is after ' + d[int(float(L[2]))]
  else :
      ch = 'the day is between  ' + d[int(float(L[0]))] + ' and ' + d[int(float(L[4]))]

  return(ch[:len(ch)])

def weather_avg_to_text(rule) :
  rule = rule.split()
  L = rule.copy()
  index = rule.index('weather_avg')
  ch = ''
  if (index == 0 )  : # on est donc dans le cas d'une régle ss la forme ..<...
     if ((L[1] == '<=') or (L[1] == '<')) :
       ch = ch + ' because the average temperature is less than ' + L[2]
     else :
       ch = ch + ' because the average temperature is more than ' + L[2]
  else :
    ch = 'the average temperature is between  ' + L[0]+ ' and ' + L[4]
  return(ch)



def day_string_to_text(rule) :
  rule = rule.split()
  L = rule.copy()
  index = rule.index('day_string')
  ch = ''
  days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
  if (index == 0 )  : # on est donc dans le cas d'une régle ss la forme ..<...
     if ((L[1] == '<=') or (L[1] == '<')) :
       ch = ch + ' because we are before ' + days[int(float(L[2]))]
     else :
       ch = ch + ' because we have exceeded ' + days[int(L[2])]
  else :
    ch = 'we are between  ' + days[int(float(L[0]))] + ' and ' + days[int(float(L[4]))]
  return(ch)

def week_of_year_to_text(rule) :
  rule = rule.split()
  L = rule.copy()
  index = rule.index('week_of_year')
  ch = ''
  if (index == 0 )  : # on est donc dans le cas d'une régle ss la forme ..<...
     if ((L[1] == '<=') or (L[1] == '<')) :
       ch = ch + ' because we did not achieve the week  ' + str(int(float(L[2]))) + ' yet'
     else :
       ch = ch + ' because we passed the week ' + str(int(float(L[2])))
  else :
    ch = 'the week is between the week ' + str(int(float(L[0])))+ ' and ' + str(int(float(L[4])))
  return(ch)



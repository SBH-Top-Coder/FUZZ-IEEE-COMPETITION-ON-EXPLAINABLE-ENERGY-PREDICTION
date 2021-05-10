import numpy as np
import pandas as pd
from tqdm import tqdm
from Utils import *
class Meter_2018:
    def __init__(self, df, date):
        self.meter_id = df.iloc[0].values[0]  # kénk m5arjo ml cons ... kén ml submission badél
        self.meter = date

    def time_features(self):
        df = self.meter
        df = df.set_index('Datetime')
        df1 = pd.DataFrame([], columns=["year", "month", "day", "hour", "minute"])
        from datetime import datetime
        for i in range(len(df.index)):
            date = datetime.strptime(df.index[i], '%Y-%m-%d %H:%M:%S')
            df_tmp = pd.DataFrame([(date.year, date.month, date.day, date.hour, date.minute)],
                                  columns=["year", "month", "day", "hour", "minute"])
            df1 = df1.append(df_tmp)
        df1 = df1.reset_index()
        df1.drop('index', axis=1, inplace=True)
        df_train = pd.concat([self.meter, df1], axis=1, ignore_index=True)
        self.meter = df_train
        self.meter.columns = ["Datetime", "year", "month", "day", "hour", "minute"]
        self.meter['season'] = self.meter['month'].apply(lambda month_number: (month_number % 12 + 3) // 3)
        self.meter['dayofyear'] = [pd.to_datetime(df.index[i]).timetuple().tm_yday for i in range(len(self.meter))]
        self.meter['day_string'] = [pd.to_datetime(self.meter['Datetime'][i]).strftime("%A") for i in
                                    range(len(self.meter))]
        self.meter['week_of_year'] = pd.to_datetime(self.meter['Datetime']).dt.weekofyear

    def weather_max(self, df):
        temp = df[df['meter_id'] == self.meter_id].transpose()[1:]
        temp.columns = ['weather']
        L = []
        for x in temp['weather']:
            for i in range(48):
                L.append(x)
        self.meter['weather_max'] = L

    def weather_min(self, df):
        temp = df[df['meter_id'] == self.meter_id].transpose()[1:]
        temp.columns = ['weather']
        L = []
        for x in temp['weather']:
            for i in range(48):
                L.append(x)
        self.meter['weather_min'] = L

    def weather_avg(self, df):
        temp = df[df['meter_id'] == self.meter_id].transpose()[1:]
        temp.columns = ['weather']
        L = []
        for x in temp['weather']:
            for i in range(48):
                L.append(x)
        self.meter['weather_avg'] = L

    def weather_cluster(self):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        weather_scaled = scaler.fit_transform(self.meter[['weather_min', 'weather_avg', 'weather_max', 'season']])
        kmeans = KMeans(n_clusters=4, max_iter=600, algorithm='auto')
        kmeans.fit(weather_scaled)
        self.meter['weather_cluster'] = kmeans.labels_

    def work_feat(self):
        self.meter["Work"] = [feature_worktime(self.meter.iloc[i]) for i in range(len(self.meter))]

    def peak_feat(self):
        self.meter["peak"] = [feature_peak(self.meter.iloc[i]) for i in range(len(self.meter))]

    def weekend_feat(self):
        self.meter["weekend"] = [feature_weekend(self.meter.iloc[i]) for i in range(len(self.meter))]

    def denya_berda(self):
        self.meter["denya_berda"] = [feature_vis1(self.meter.iloc[i]) for i in range(len(self.meter))]

    def add_info(self, add):
        self.meter['dwelling_type'] = add[add['meter_id'] == self.meter_id]['dwelling_type'].values[0]
        self.meter['num_bedrooms'] = add[add['meter_id'] == self.meter_id]['num_bedrooms'].values[0]

    def feature_inutile(self):
        self.meter.drop(['Datetime', 'year', 'day', 'minute'], axis=1, inplace=True)  # à priori inutile

    def casting(self):
        self.meter['hour'] = self.meter['hour'].astype('int64')
        self.meter['month'] = self.meter['month'].astype('int64')

    def mapping(self):
        self.meter['Work'] = self.meter['Work'].map({'Worktime': 1, 'NonWorkTime': 0})
        self.meter['peak'] = self.meter['peak'].map({'NonPeak': 0, 'Peak': 1})
        self.meter['weekend'] = self.meter['weekend'].map({'NonWeekend': 0, 'Weekend': 1})
        self.meter['denya_berda'] = self.meter['denya_berda'].map({'bard': 1, '3adi': 0})
        self.meter['dwelling_type'] = self.meter['dwelling_type'].map(
            {'_': -1, 'semi_detached_house': 1, 'detached_house': 2,
             'terraced_house': 3, 'bungalow': 4, 'flat': 5})

    def encoding(self):
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        self.meter['day_string'] = le.fit_transform(self.meter['day_string'])


import warnings

from tqdm import tqdm

from Meter_2017 import Meter_2017
from Meter_2018 import Meter_2018
from Train import Train
from Explainer import  Explainer
from Utils import *

warnings.filterwarnings('ignore')
import xgboost as xgb
model = xgb.XGBRegressor()
parameters = {'max_depth'         : [50,100,200,500,1000,2000,5000,10000,20000,50000,100000]
                  'learning_rate' : [0,005,0.01,0.02,0.05,0.1,0.15,0.2,0.5],      # les paramétres li t7éb tlawéj 3la a7san combinaison binéthom
                  'n_estimators'    : [5000,10000,20000,50000,100000,500000,1000000,2000000,5000000]
                 }


def main():
    # loading All needed data
    print('loading')
    cons = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\consumption.csv")
    weather_max = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\weather-max.csv")
    weather_min = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\weather-min.csv")
    weather_avg = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\weather-avg.csv")
    add = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\addInfo.csv")
    sub = load(r"C:\Users\Asus\Desktop\PFE_Partie2\data\data\sample_submission.csv")
    # filling nans
    for x in tqdm(cons.columns) :
      impute_na(cons,x)
    print('filling')
    for x in tqdm(weather_min.columns):

        impute_na(weather_min, x)
        impute_na(weather_avg, x)
        impute_na(weather_max, x)
    # ta74ir add
    print("ta74ir add")
    add = prepare_add(cons, add)
    # ta74ir train
    for i in range(3248):
        print('itération',i)
        meter_2017 = Meter_2017(pd.DataFrame(cons.iloc[i]))
        meter_2017.drop_duplicate()
        meter_2017.time_features()
        meter_2017.weather_min(weather_min)
        meter_2017.weather_avg(weather_avg)
        meter_2017.weather_max(weather_max)
        meter_2017.weather_cluster()
        meter_2017.work_feat()
        meter_2017.peak_feat()
        meter_2017.weekend_feat()
        meter_2017.denya_berda()
        meter_2017.add_info(add)
        meter_2017.feature_inutile()
        meter_2017.casting()
        meter_2017.mapping()
        meter_2017.encoding()
        if (i == 0):  # kén na3mlou df1 = ... twali kolha object .... bl façon héthi on évite float
            train = meter_2017.meter
        else:
            train = pd.concat([train, meter_2017.meter])
    train.to_csv('Train.csv')

    print('preparing date')
    date = []
    for x in cons.iloc[0][1:].index:
        L = list(x)
        L[3] = '8'
        date.append("".join(L))
    date = pd.DataFrame(date)
    date.columns = ['Datetime']
    print(date)

    print('preparing test set')
    for i in range(3248):
        print('itération',i)
        meter_2018= Meter_2018(pd.DataFrame(cons.iloc[i]),date)
        meter_2018.time_features()
        meter_2018.weather_min(weather_min)
        meter_2018.weather_avg(weather_avg)
        meter_2018.weather_max(weather_max)
        meter_2018.weather_cluster()
        meter_2018.work_feat()
        meter_2018.peak_feat()
        meter_2018.weekend_feat()
        meter_2018.denya_berda()
        meter_2018.add_info(add)
        meter_2018.feature_inutile()
        meter_2018.casting()
        meter_2018.mapping()
        meter_2018.encoding()
        if (i == 0):  # kén na3mlou df1 = ... twali kolha object .... bl façon héthi on évite float
            test = meter_2018.meter
        else:
            test = pd.concat([test, meter_2018.meter])
    test.to_csv('Test.csv')


    print('train')
    training = Train(train, test)
    training.model(model)
    training.parameters(parameters)
    training.GridSearch()
    training.Gridfit()
    training.finalfit()
    pred = training.predict()
    print('explainability ')
    explainer = Explainer(train.drop('consommation',axis=1),test)

    L = []
    for j in range(3248):
        explainer.explain_instance(1000+17520*j,training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(2000+17520*j,training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(3000 + 17520 * j,training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(4500 + 17520 * j,training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(6000 + 17520 * j, training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(8000 + 17520 * j, training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(9000 + 17520 * j, training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(11000 + 17520 * j, training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(12000 + 17520 * j, training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(1400 + 17520 * j, training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(15000 + 17520 * j, training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

        explainer.explain_instance(17000 + 17520 * j, training.grid.best_estimator_)
        L.append(explainer.exp.as_list())

    Features = ['denya_berda', 'weather_cluster', 'season', 'month', 'dwelling_type', 'num_bedrooms']
    Months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
              'October','November', 'December']
    Explanations = []
    for i in range(38976):  # bl chhar bl chhar
        K = L[i]  # nchédo explication
        ch = 'Explanation of the consumption of ' + Months[(i%12)]  +  ' : '
        for f in Features:  # bl feature bl feature
            j = 0
            position = 0
            index = K[j][0].find(f)
            while (index == -1) and (j < len(K)):
                index = K[j][0].find(f)
                position = j
                j = j + 1
            if (index != -1):
                rule = K[position][0]
                if (f == 'denya_berda'):
                    ch = ch + denya_berda_to_text(rule)
                elif (f == 'weather_cluster'):
                    ch = ch + weather_cluster_to_text(rule)
                elif (f == 'season'):
                    ch = ch + season_to_text(rule)
                elif (f == 'month'):
                    ch = ch + month_to_text(rule)
                elif (f == 'dwelling_type'):
                    ch = ch + dwelling_type_to_text(rule)
                elif (f == 'num_bedrooms'):
                    ch = ch + num_bedrooms_to_text(rule)
        Explanations.append(ch)
    submit_file = submit(sub.columns, cons, pred,Explanations)
    submit_file.to_csv("Explanations.csv")





if __name__ == "__main__":
    main()
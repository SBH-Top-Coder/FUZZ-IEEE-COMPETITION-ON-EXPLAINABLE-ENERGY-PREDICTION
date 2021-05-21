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
parameters = {'max_depth'         : [1,2],
                  'learning_rate' : [0.1,0.15],      # les paramétres li t7éb tlawéj 3la a7san combinaison binéthom
                  'n_estimators'    : [1,2]
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
    # variables global
    janv = cons.loc[:, '2017-01-01 00:00:00':'2017-01-31 23:30:00']
    janv_moy = pd.DataFrame(janv.dropna(axis=0, how='any').mean(axis=0))
    janv_moy.columns = ['consommation']
    fev = cons.loc[:, '2017-02-01 00:00:00':'2017-02-28 23:30:00']
    fev_moy = pd.DataFrame(fev.dropna(axis=0, how='any').mean(axis=0))
    fev_moy.columns = ['consommation']
    mars = cons.loc[:, '2017-01-03 00:00:00':'2017-03-31 23:30:00']
    mars_moy = pd.DataFrame(mars.dropna(axis=0, how='any').mean(axis=0))
    mars_moy.columns = ['consommation']
    avril = cons.loc[:, '2017-04-01 00:00:00':'2017-04-30 23:30:00']
    avril_moy = pd.DataFrame(avril.dropna(axis=0, how='any').mean(axis=0))
    avril_moy.columns = ['consommation']
    may = cons.loc[:, '2017-05-01 00:00:00':'2017-05-31 23:30:00']
    may_moy = pd.DataFrame(may.dropna(axis=0, how='any').mean(axis=0))
    may_moy.columns = ['consommation']
    juin = cons.loc[:, '2017-06-01 00:00:00':'2017-06-30 23:30:00']
    juin_moy = pd.DataFrame(juin.dropna(axis=0, how='any').mean(axis=0))
    juin_moy.columns = ['consommation']
    juil = cons.loc[:, '2017-07-01 00:00:00':'2017-07-31 23:30:00']
    juil_moy = pd.DataFrame(juil.dropna(axis=0, how='any').mean(axis=0))
    juil_moy.columns = ['consommation']
    aout = cons.loc[:, '2017-08-01 00:00:00':'2017-08-31 23:30:00']
    aout_moy = pd.DataFrame(aout.dropna(axis=0, how='any').mean(axis=0))
    aout_moy.columns = ['consommation']
    sept = cons.loc[:, '2017-09-01 00:00:00':'2017-09-30 23:30:00']
    sept_moy = pd.DataFrame(sept.dropna(axis=0, how='any').mean(axis=0))
    sept_moy.columns = ['consommation']
    oct = cons.loc[:, '2017-10-01 00:00:00':'2017-10-31 23:30:00']
    oct_moy = pd.DataFrame(oct.dropna(axis=0, how='any').mean(axis=0))
    oct_moy.columns = ['consommation']
    nov = cons.loc[:, '2017-11-01 00:00:00':'2017-11-30 23:30:00']
    nov_moy = pd.DataFrame(nov.dropna(axis=0, how='any').mean(axis=0))
    nov_moy.columns = ['consommation']
    dec = cons.loc[:, '2017-12-01 00:00:00':'2017-12-31 23:30:00']
    dec_moy = pd.DataFrame(dec.dropna(axis=0, how='any').mean(axis=0))
    dec_moy.columns = ['consommation']
    # filling nans
    print('filling cons ')
    impute_na (cons, janv_moy, fev_moy, mars_moy, avril_moy, may_moy, juin_moy, juil_moy, aout_moy, sept_moy, oct_moy, nov_moy,dec_moy)
    print('filling weather')
    for x in tqdm(weather_min.columns):

        random_sampling(weather_min, x)
        random_sampling(weather_avg, x)
        random_sampling(weather_max, x)

    # ta74ir add
    print("ta74ir add")
    add = prepare_add_wissem (cons,add)
    add = prepare_add_semi (cons,add)
    add = fill_add_kmeans(add, cons)
#     add = fill_add_distance(add, cons)
    # ta74ir train
    for i in range(3):
        print('itération',i)
        meter_2017 = Meter_2017(pd.DataFrame(cons.iloc[i]))
        # impute_na(meter_2017.meter, janv_moy, fev_moy, mars_moy, avril_moy, may_moy, juin_moy, juil_moy, aout_moy, sept_moy,oct_moy, nov_moy, dec_moy)
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
    for i in range(1):
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
    for j in tqdm(range(1)):
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
    # Explanations_month_ = []
    # for i in range(12):  # bl chhar bl chhar
    #     K = L[i]  # nchédo explication
    #     ch = 'Explanation of the consumption of ' + Months[(i%12)]  +  ' : '
    #     for f in Features:  # bl feature bl feature
    #         j = 0
    #         position = 0
    #         index = K[j][0].find(f)
    #         while (index == -1) and (j < len(K)):
    #             index = K[j][0].find(f)
    #             position = j
    #             j = j + 1
    #         if (index != -1):
    #             rule = K[position][0]
    #             if (f == 'denya_berda'):
    #                 ch = ch + denya_berda_to_text(rule)
    #             elif (f == 'weather_cluster'):
    #                 ch = ch + weather_cluster_to_text(rule)
    #             elif (f == 'season'):
    #                 ch = ch + season_to_text(rule)
    #             elif (f == 'month'):
    #                 ch = ch + month_to_text(rule)
    #             elif (f == 'dwelling_type'):
    #                 ch = ch + dwelling_type_to_text(rule)
    #             elif (f == 'num_bedrooms'):
    #                 ch = ch + num_bedrooms_to_text(rule)
    #     Explanations_month.append(ch)
    # submit_file_month = submit_month(sub.columns, cons, pred,Explanations_month,weather)
    # submit_file_month.to_csv("Explanations.csv")

    # Explanations day
    #
    # L = []
    # for j in tqdm(range(1)):
    #     for k  in range(365) :
    #         explainer.explain_instance(48*k+17520*j,training.grid.best_estimator_)
    #         L.append(explainer.exp.as_list())

    # Explanations_day_ = []
    # for i in range(365):  # bl nhar bl nhar ...
    #     K = L[i]  # nchédo explication
    #     ch = 'Explanation of the consumption of ' + cons.columns[(1+i*48)].split()[0] + ' : '
    #     for f in Features:  # bl feature bl feature
    #         j = 0
    #         position = 0
    #         index = K[j][0].find(f)
    #         while (index == -1) and (j < len(K)):
    #             index = K[j][0].find(f)
    #             position = j
    #             j = j + 1
    #         if (index != -1):
    #             rule = K[position][0]
    #             if (f == 'denya_berda'):
    #                 ch = ch + denya_berda_to_text(rule)
    #             elif (f == 'weather_cluster'):
    #                 ch = ch + weather_cluster_to_text(rule)
    #             elif (f == 'season'):
    #                 ch = ch + season_to_text(rule)
    #             elif (f == 'month'):
    #                 ch = ch + month_to_text(rule)
    #             elif (f == 'dwelling_type'):
    #                 ch = ch + dwelling_type_to_text(rule)
    #             elif (f == 'num_bedrooms'):
    #                 ch = ch + num_bedrooms_to_text(rule)
    #             elif (f == 'weather_avg'):
    #                 ch = ch + weather_avg_to_text(rule)
    #             elif (f == 'day_string'):
    #                 ch = ch + day_string_to_text(rule)
    #             elif (f == 'dayofyear'):
    #                 ch = ch + dayofyear_to_text(rule)
    #
    #     Explanations_day.append(ch)
    # submit_day(cons, pred, Explanations_day)

    # Explanations week

    L = []
    for j in tqdm(range(1)):
        for k  in range(52) :
            explainer.explain_instance(336*k+17520*j,training.grid.best_estimator_)
            L.append(explainer.exp.as_list())

    Explanations_week_ = []
    for i in tqdm(range(52)):  # bl nhar bl nhar ...
        K = L[i]  # nchédo explication
        ch = 'Explanation of the consumption of ' + cons.columns[(1 + i * 48)].split()[0] + ' : '
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
                elif (f == 'weather_avg'):
                    ch = ch + weather_avg_to_text(rule)
                elif (f == 'day_string'):
                    ch = ch + day_string_to_text(rule)
                elif (f == 'dayofyear'):
                    ch = ch + dayofyear_to_text(rule,day_to_int(cons))
                elif (f == 'week_of_year') :
                    ch = ch + week_of_year_to_text(rule)


        Explanations_week_.append(ch)
    submit_week(cons, pred, Explanations_week)


if __name__ == "__main__":
    main()

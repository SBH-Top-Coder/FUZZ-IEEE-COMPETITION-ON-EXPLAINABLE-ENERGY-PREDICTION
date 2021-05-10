from lime.lime_tabular import LimeTabularExplainer
import numpy as np

class Explainer :
    def __init__(self,X_train,X_test):
        self.X_train = X_train
        self.X_test = X_test
        self.explainer = LimeTabularExplainer(
    training_data=np.asarray(self.X_train), feature_names=self.X_train.columns,class_names=['consommation'],mode='regression' )

    def explain_instance (self,i,model) :
        self.exp = self.explainer.explain_instance(
            data_row= self.X_test.iloc[i].values,
            predict_fn=model.predict)


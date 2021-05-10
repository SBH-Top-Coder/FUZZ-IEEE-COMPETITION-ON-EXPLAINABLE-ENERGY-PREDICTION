import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
class Train :
  def __init__(self,train,test) :
    self.data = train
    self.X_train = train.drop('consommation',axis=1)
    self.y_train = train['consommation']
    self.X_test = test
  def model (self,model) :
    self.model = model
  def parameters (self, parameters) :
    self.parameters = parameters
  def GridSearch (self) :
    self.grid = GridSearchCV(estimator=self.model, param_grid = self.parameters, cv = 5, n_jobs=-1)
  def Gridfit (self) :
    self.grid.fit(np.asarray(self.X_train),np.asarray(self.y_train))
  def finalfit (self) :
    self.grid.best_estimator_.fit(np.asarray(self.X_train),np.asarray(self.y_train))
  def predict(self):
    self.pred = self.grid.best_estimator_.predict(np.asarray(self.X_test))
    return (self.pred)
  def score(self,label):
    from sklearn.metrics import mean_absolute_error
    mean_absolute_error(label,self.pred)





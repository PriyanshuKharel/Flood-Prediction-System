#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class LinearRegression():
    
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        
    def fit(self, X, Y):
        self.m, self.n = X.shape
        
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        
        for i in range(self.no_of_iterations):
            self.update_weights()
            
    def update_weights(self):
        
        Y_prediction = self.predict(self.X)
        
        dw = - (2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        
        db = - 2 * np.sum(self.Y - Y_prediction) / self.m
        
        self.w = self.w - self.learning_rate*dw
        self.b = self.b - self.learning_rate*db
        
        
    def predict(self, X):
        
        y_pred = X.dot(self.w) + self.b
        y_pred = np.where( y_pred > 6, 1, 0)
        return y_pred
            
        


# In[ ]:





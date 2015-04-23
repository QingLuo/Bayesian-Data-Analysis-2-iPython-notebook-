
from scipy import *
from univariate_bayes import UnivariateBayesianInference
import operator
import scipy
import numpy as np

class  CauchyLocationInference(UnivariateBayesianInference):
    """
    Cauchy Bayesian inference 
    """
    def __init__(self, d , data , prior=1.,na=200):
        """
        d : scale 

        data: support of location
        """
        self.d, self.data = d, data
        self.na = na
        lowerb=-10
        upperb=10
        self.locs = linspace(lowerb,upperb,na)

#initialize the base class
        super(CauchyLocationInference, self).__init__(self.locs, prior, self.lfunc)
        
#likelihood function. 
    def lfunc(self, x):
        production=1
        for i in range(1,len(self.data)):
            production=production*((self.data[i-1]-x)**2+self.d**2)    
        return (1/production)*(self.d/3.14159)**(len(self.data))

    
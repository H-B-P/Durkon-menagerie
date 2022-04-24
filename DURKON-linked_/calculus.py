import pandas as pd
import numpy as np

def Gauss_grad(pred,act):
 return 2*(pred-act)

def Poisson_grad(pred,act):
 return (pred-act)/pred

def Gamma_grad(pred,act):
 return (pred-act)/(pred*pred)

def Logit_grad(pred,act):
 return (pred-act)/(pred*(1-pred))



def Unity_link(x):
 return x

def Unity_link_grad(x):
 return 1

def Root_link(x):
 return x*x

def Root_link_grad(x):
 return 2*x

def Log_link(x):
 return np.exp(x)

def Log_link_grad(x):
 return np.exp(x)


def Logit_link(x):
 return 1/(1+np.exp(-x))

def Logit_link_grad(x):
 return np.exp(-x)/((1+np.exp(-x))**2)
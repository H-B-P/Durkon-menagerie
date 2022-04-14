import pandas as pd
import numpy as np

def Gauss_grad(pred,act):
 return 2*(pred-act)

def Poisson_grad(pred,act):
 return (pred-act)/pred

def Gamma_grad(pred,act):
 return (pred-act)/(pred*pred)


def Unity_link(x):
 return x

def Unity_link_grad(x):
 return 1

def Log_link(x):
 return np.exp(x)

def Log_link_grad(x):
 return np.exp(x)



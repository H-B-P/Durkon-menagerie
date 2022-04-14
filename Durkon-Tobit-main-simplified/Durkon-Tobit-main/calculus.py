import pandas as pd
import numpy as np

import math

import scipy
from scipy.special import erf

def Gauss_grad(pred,act):
 return 2*(pred-act)

def Gauss_hess(pred,act):
 return 2

def Poisson_grad(pred,act):
 return (pred-act)/pred

def Poisson_hess(pred,act):
 return act/(pred*pred)

def Gamma_grad(pred,act):
 return (pred-act)/(pred*pred)

def Gamma_hess(pred,act):
 return (2*act-pred)/(pred*pred*pred)

def gnormal_u_diff(y, u, p):
 return y*(y-u)/((p**2)*(u**3)) - 1/u

def gnormal_p_diff(y, u, p):
 return (y-u)**2/((p**3)*(u**2)) - 1/p

def PDF(y, u, p):
 return np.exp(-0.5*((y-u)/(p*u))**2) / (p*u*math.sqrt(2*math.pi))

def CDF(y, u, p):
 return 0.5*(1 - erf((y-u)/(p*u*math.sqrt(2))))

def u_diff_censored(y, u, p):
 return (y/u)*PDF(y,u,p)/CDF(y,u,p)

def p_diff_censored(y, u, p):
 return ((y-u)/p)*PDF(y,u,p)/CDF(y,u,p)


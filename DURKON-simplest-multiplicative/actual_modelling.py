import pandas as pd
import numpy as np
import math
import copy
import time

import util
import apply_model
import calculus

def produce_cont_relevances(inputDf, model, col):
 reles=np.zeros((len(model["conts"][col])+1,len(inputDf)))
 
 reles[0][(inputDf[col]<=model["conts"][col][0][0])] = 1 #d(featpred)/d(pt)
 for i in range(len(model["conts"][col])-1):
  x = inputDf[col]
  x1 = model["conts"][col][i][0]
  x2 = model["conts"][col][i+1][0]
  subset = (x>=x1) & (x<=x2)
  #print(reles[subset][:,1])
  reles[i][subset] = (x2 - x[subset])/(x2 - x1) #d(featpred)/d(pt)
  reles[i+1][subset] = (x[subset] - x1)/(x2 - x1) #d(featpred)/d(pt)
 reles[-2][(inputDf[col]>=model["conts"][col][-1][0])] = 1 #d(featpred)/d(pt)
 
 reles[-1] = 1 - np.sum(reles, axis=0)
 return np.transpose(reles) #roundabout way of doing this but the rest of the function doesn't flow as naturally if x and y don't switch places

def produce_cont_relevances_dict(inputDf, model):
 opDict = {}
 
 for col in model["conts"]:
  opDict[col]=produce_cont_relevances(inputDf, model, col)
 
 return opDict

def produce_cat_relevances(inputDf, model, col):
 reles=np.zeros((len(model["cats"][col]["uniques"])+1,len(inputDf)))
 
 skeys = apply_model.get_sorted_keys(model, col)
 for i in range(len(skeys)):
  reles[i][inputDf[col].isin([skeys[i]])] = 1 #d(featpred)/d(pt)
 reles[-1][~inputDf[col].isin(skeys)] = 1 #d(featpred)/d(pt)
 
 return np.transpose(reles) #roundabout way of doing this but the rest of the function doesn't flow as naturally if x and y don't switch places

def produce_cat_relevances_dict(inputDf, model):
 opDict = {}
 
 for col in model["cats"]:
  opDict[col]=produce_cat_relevances(inputDf, model, col)
 
 return opDict

def sum_and_listify_matrix(a):
 return np.array(sum(a)).tolist()

def produce_total_relevances_dict(contReleDict, catReleDict):
 op = {"conts":{},"cats":{}}
 for col in contReleDict:
  print(sum(contReleDict[col]))
  op["conts"][col] = sum_and_listify_matrix(contReleDict[col])
 for col in catReleDict:
  op["cats"][col] = sum_and_listify_matrix(catReleDict[col])
 print(op)
 return op

def produce_wReleDict(releDict, w):
 wReleDict = {}
 for col in releDict:
  wReleDict[col]=w*releDict[col]
 return wReleDict

def train_model(inputDf, target, nrounds, lr, startingModel, weights=None, grad=calculus.Gauss_grad):
 
 model = copy.deepcopy(startingModel)
 
 if weights==None:
  weights = np.ones(len(inputDf))
 w = np.array(np.transpose(np.matrix(weights)))
 sw = sum(weights)
 
 cord = produce_cont_relevances_dict(inputDf,model)
 card = produce_cat_relevances_dict(inputDf,model)
 tord = produce_total_relevances_dict(cord, card)
 
 cowrd = produce_wReleDict(cord, w)
 cawrd = produce_wReleDict(card, w)
 towrd = produce_total_relevances_dict(cowrd, cawrd)
 
 for i in range(nrounds):
  
  print("epoch: "+str(i+1)+"/"+str(nrounds))
  apply_model.explain(model)
  
  print("initial pred and effect-gathering")
  
  contEffects = apply_model.get_effects_of_cont_cols_from_relevance_dict(cord,model)
  catEffects = apply_model.get_effects_of_cat_cols_from_relevance_dict(card,model)
  
  pred = apply_model.pred_from_effects(model["BASE_VALUE"], len(inputDf), contEffects, catEffects)
  
  gradient = grad(pred, np.array(inputDf[target])) #d(Loss)/d(pred)
  
  print("adjust conts")
  
  for col in model["conts"]:
   
   effectOfCol = contEffects[col]
   
   peoc = pred/effectOfCol #d(pred)/d(featpred)
   
   finalGradients = np.matmul(np.array(peoc*gradient),cowrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
   
   for k in range(len(model['conts'][col])):
    totRele = towrd["conts"][col][k]
    if totRele>0:
     model["conts"][col][k][1] -= finalGradients[k]*lr/totRele #and not /sw
       
      
   print("adjust cats")
   
  for col in model["cats"]:
   
   effectOfCol = catEffects[col]
   
   peoc = pred/effectOfCol #d(pred)/d(featpred)
   
   finalGradients = np.matmul(np.array(peoc*gradient),cawrd[col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
   
   skeys = apply_model.get_sorted_keys(model, col)
   
   #all the uniques . . .
   for k in range(len(skeys)):
    totRele = towrd["cats"][col][k]
    if totRele>0:
     model["cats"][col]["uniques"][skeys[k]] -= finalGradients[k]*lr/totRele #and not /sw
   
   # . . . and "OTHER"
   totRele = towrd[m]["cats"][col][-1]
   if totRele>0:
    model["cats"][col]["OTHER"] -= finalGradients[-1]*lr/totRele #and not /sw
 return model

if __name__ == '__main__':
 df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8,9],"y":[1,2,3,4,5,6,7,8,9]})
 model = {"BASE_VALUE":3.0,"conts":{"x":[[1,5],[5,5],[9,5]]}, "cats":[]}
 newModel = train_model(df, "y",100, 0.1, model,weights=[1,2,3,4,5,6,7,8,9])
 apply_model.explain(newModel)

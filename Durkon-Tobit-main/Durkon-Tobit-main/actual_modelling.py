import pandas as pd
import numpy as np
import math
import copy
import time

import util
import apply_model
import calculus

def produce_cont_relevances(inputDf, model, col):
 reles=np.zeros((len(model["conts"][col]),len(inputDf)))
 
 reles[0][(inputDf[col]<=model["conts"][col][0][0])] = 1 #d(featpred)/d(pt)
 for i in range(len(model["conts"][col])-1):
  x = inputDf[col]
  x1 = model["conts"][col][i][0]
  x2 = model["conts"][col][i+1][0]
  subset = (x>=x1) & (x<=x2)
  #print(reles[subset][:,1])
  reles[i][subset] = (x2 - x[subset])/(x2 - x1) #d(featpred)/d(pt)
  reles[i+1][subset] = (x[subset] - x1)/(x2 - x1) #d(featpred)/d(pt)
 reles[-1][(inputDf[col]>=model["conts"][col][-1][0])] = 1 #d(featpred)/d(pt)
 
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

def train_model(inputDf, inputDfC, target, nrounds, lrsU, lrsP, startingModelsU, startingModelsP, weights=None, weightsC=None, gradU=calculus.gnormal_u_diff, gradP=calculus.gnormal_p_diff, gradUC=calculus.u_diff_censored, gradPC=calculus.p_diff_censored):
 
 modelsU = copy.deepcopy(startingModelsU)
 modelsP = copy.deepcopy(startingModelsP)
 
 if weights==None:
  weights = np.ones(len(inputDf))
 w = np.array(np.transpose(np.matrix(weights)))
 sw = sum(weights)
 
 if weightsC==None:
  weightsC = np.ones(len(inputDfC))
 wC = np.array(np.transpose(np.matrix(weightsC)))
 swC = sum(weightsC)
 
 contReleDictListU=[]
 catReleDictListU=[]
 totReleDictListU=[]
 
 contWReleDictListU=[]
 catWReleDictListU=[]
 totWReleDictListU=[]
 
 contReleDictListP=[]
 catReleDictListP=[]
 totReleDictListP=[]
 
 contWReleDictListP=[]
 catWReleDictListP=[]
 totWReleDictListP=[]
 
 contReleDictListUC=[]
 catReleDictListUC=[]
 totReleDictListUC=[]
 
 contWReleDictListUC=[]
 catWReleDictListUC=[]
 totWReleDictListUC=[]
 
 contReleDictListPC=[]
 catReleDictListPC=[]
 totReleDictListPC=[]
 
 contWReleDictListPC=[]
 catWReleDictListPC=[]
 totWReleDictListPC=[]
 
 print("initial relevances setup")
 
 for model in modelsU:
  cord = produce_cont_relevances_dict(inputDf,model)
  card = produce_cat_relevances_dict(inputDf,model)
  
  contReleDictListU.append(cord)
  catReleDictListU.append(card)
  totReleDictListU.append(produce_total_relevances_dict(cord, card))
  
  cowrd = produce_wReleDict(cord, w)
  cawrd = produce_wReleDict(card, w)
  
  contWReleDictListU.append(cowrd)
  catWReleDictListU.append(cawrd)
  totWReleDictListU.append(produce_total_relevances_dict(cowrd, cawrd))
 
 for model in modelsP:
  cord = produce_cont_relevances_dict(inputDf,model)
  card = produce_cat_relevances_dict(inputDf,model)
  
  contReleDictListP.append(cord)
  catReleDictListP.append(card)
  totReleDictListP.append(produce_total_relevances_dict(cord, card))
  
  cowrd = produce_wReleDict(cord, w)
  cawrd = produce_wReleDict(card, w)
  
  contWReleDictListP.append(cowrd)
  catWReleDictListP.append(cawrd)
  totWReleDictListP.append(produce_total_relevances_dict(cowrd, cawrd))
 
 for model in modelsU:
  cord = produce_cont_relevances_dict(inputDfC,model)
  card = produce_cat_relevances_dict(inputDfC,model)
  
  contReleDictListUC.append(cord)
  catReleDictListUC.append(card)
  totReleDictListUC.append(produce_total_relevances_dict(cord, card))
  
  cowrd = produce_wReleDict(cord, wC)
  cawrd = produce_wReleDict(card, wC)
  
  contWReleDictListUC.append(cowrd)
  catWReleDictListUC.append(cawrd)
  totWReleDictListUC.append(produce_total_relevances_dict(cowrd, cawrd))
 
 for model in modelsP:
  cord = produce_cont_relevances_dict(inputDfC,model)
  card = produce_cat_relevances_dict(inputDfC,model)
  
  contReleDictListPC.append(cord)
  catReleDictListPC.append(card)
  totReleDictListPC.append(produce_total_relevances_dict(cord, card))
  
  cowrd = produce_wReleDict(cord, wC)
  cawrd = produce_wReleDict(card, wC)
  
  contWReleDictListPC.append(cowrd)
  catWReleDictListPC.append(cawrd)
  totWReleDictListPC.append(produce_total_relevances_dict(cowrd, cawrd))
 
 for i in range(nrounds):
  
  #First, do everything for the uncensored df . . .
  
  print("epoch: "+str(i+1)+"/"+str(nrounds))
  print('U')
  for model in modelsU:
   apply_model.explain(model)
  
  print('P')
  for model in modelsP:
   apply_model.explain(model)
  
  print("initial pred and effect-gathering")
  
  predsU=[]
  overallPredU=pd.Series([0]*len(inputDf))
  contEffectsListU=[]
  catEffectsListU=[]
  
  for m in range(len(modelsU)):
   
   contEffectsU = apply_model.get_effects_of_cont_cols_from_relevance_dict(contReleDictListU[m],modelsU[m])
   contEffectsListU.append(contEffectsU)
   catEffectsU = apply_model.get_effects_of_cat_cols_from_relevance_dict(catReleDictListU[m],modelsU[m])
   catEffectsListU.append(catEffectsU)
   
   predU = apply_model.pred_from_effects(modelsU[m]["BASE_VALUE"], len(inputDf), contEffectsU, catEffectsU)
   predsU.append(predU)
   overallPredU += predU
  
  predsP=[]
  overallPredP=pd.Series([0]*len(inputDf))
  contEffectsListP=[]
  catEffectsListP=[]
  
  for m in range(len(modelsP)):
   
   contEffectsP = apply_model.get_effects_of_cont_cols_from_relevance_dict(contReleDictListP[m],modelsP[m])
   contEffectsListP.append(contEffectsP)
   catEffectsP = apply_model.get_effects_of_cat_cols_from_relevance_dict(catReleDictListP[m],modelsP[m])
   catEffectsListP.append(catEffectsP)
   
   predP = apply_model.pred_from_effects(modelsP[m]["BASE_VALUE"], len(inputDf), contEffectsP, catEffectsP)
   predsP.append(predP)
   overallPredP += predP
  
  gradientU = -gradU(np.array(inputDf[target]), overallPredU, overallPredP) #d(Loss)/d(predU)
  gradientP = -gradP(np.array(inputDf[target]), overallPredU, overallPredP) #d(Loss)/d(predP)
  
  for m in range(len(modelsU)):
   
   model=modelsU[m]
   pred=predsU[m]
   
   print("adjust conts")
   
   for col in model["conts"]:
    
    effectOfCol = contEffectsListU[m][col]
    
    peoc = predU/effectOfCol #d(predU)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradientU),contWReleDictListU[m][col]) #d(Loss)/d(pt) = d(Loss)/d(predU) * d(predU)/d(featpred) * d(featpred)/d(pt)
    
    for k in range(len(finalGradients)):
     totRele = totWReleDictListU[m]["conts"][col][k] + totWReleDictListUC[m]["conts"][col][k]
     if totRele>0:
      modelsU[m]["conts"][col][k][1] -= finalGradients[k]*lrsU[m]/totRele #and not /(sw+swC)
      
   print("adjust cats")
   
   for col in model["cats"]:
    
    effectOfCol = catEffectsListU[m][col]
    
    peoc = predU/effectOfCol #d(predU)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradientU),catWReleDictListU[m][col]) #d(Loss)/d(pt) = d(Loss)/d(predU) * d(predU)/d(featpred) * d(featpred)/d(pt)
    
    skeys = apply_model.get_sorted_keys(model, col)
    
    #all the uniques . . .
    for k in range(len(skeys)):
     totRele = totWReleDictListU[m]["cats"][col][k] + totWReleDictListUC[m]["cats"][col][k]
     if totRele>0:
      modelsU[m]["cats"][col]["uniques"][skeys[k]] -= finalGradients[k]*lrsU[m]/totRele #and not /(sw+swC)
    
    # . . . and "OTHER"
    totRele = totWReleDictListU[m]["cats"][col][-1] + totWReleDictListUC[m]["cats"][col][-1]
    if totRele>0:
     modelsU[m]["cats"][col]["OTHER"] -= finalGradients[-1]*lrsU[m]/totRele #and not /(sw+swC)
 
  for m in range(len(modelsP)):
   
   model=modelsP[m]
   pred=predsP[m]
   
   print("adjust conts")
   
   for col in model["conts"]:
    
    effectOfCol = contEffectsListP[m][col]
    
    peoc = predP/effectOfCol #d(predP)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradientP),contWReleDictListP[m][col]) #d(Loss)/d(pt) = d(Loss)/d(predP) * d(predP)/d(featpred) * d(featpred)/d(pt)
    
    for k in range(len(finalGradients)):
     totRele = totWReleDictListP[m]["conts"][col][k] + totWReleDictListPC[m]["conts"][col][k]
     if totRele>0:
      modelsP[m]["conts"][col][k][1] -= finalGradients[k]*lrsP[m]/totRele #and not /(sw+swC)
      
   print("adjust cats")
   
   for col in model["cats"]:
    
    effectOfCol = catEffectsListP[m][col]
    
    peoc = predP/effectOfCol #d(predP)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradientP),catWReleDictListP[m][col]) #d(Loss)/d(pt) = d(Loss)/d(predP) * d(predP)/d(featpred) * d(featpred)/d(pt)
    
    skeys = apply_model.get_sorted_keys(model, col)
    
    #all the uniques . . .
    for k in range(len(skeys)):
     totRele = totWReleDictListP[m]["cats"][col][k] + totWReleDictListPC[m]["cats"][col][k]
     if totRele>0:
      modelsP[m]["cats"][col]["uniques"][skeys[k]] -= finalGradients[k]*lrsP[m]/totRele #and not /(sw+swC)
    
    # . . . and "OTHER"
    totRele = totWReleDictListP[m]["cats"][col][-1] + totWReleDictListPC[m]["cats"][col][-1]
    if totRele>0:
     modelsP[m]["cats"][col]["OTHER"] -= finalGradients[-1]*lrsP[m]/totRele #and not /(sw+swC)
  
  # . . . and then again for the censored
  
  print("initial pred and effect-gathering")
  
  predsUC=[]
  overallPredUC=pd.Series([0]*len(inputDfC))
  contEffectsListUC=[]
  catEffectsListUC=[]
  
  for m in range(len(modelsU)):
   
   contEffectsUC = apply_model.get_effects_of_cont_cols_from_relevance_dict(contReleDictListUC[m],modelsU[m])
   contEffectsListUC.append(contEffectsUC)
   catEffectsUC = apply_model.get_effects_of_cat_cols_from_relevance_dict(catReleDictListUC[m],modelsU[m])
   catEffectsListUC.append(catEffectsUC)
   
   predUC = apply_model.pred_from_effects(modelsU[m]["BASE_VALUE"], len(inputDfC), contEffectsUC, catEffectsUC)
   predsUC.append(predUC)
   overallPredUC += predUC
  
  predsPC=[]
  overallPredPC=pd.Series([0]*len(inputDfC))
  contEffectsListPC=[]
  catEffectsListPC=[]
  
  for m in range(len(modelsP)):
   
   contEffectsPC = apply_model.get_effects_of_cont_cols_from_relevance_dict(contReleDictListPC[m],modelsP[m])
   contEffectsListPC.append(contEffectsPC)
   catEffectsPC = apply_model.get_effects_of_cat_cols_from_relevance_dict(catReleDictListPC[m],modelsP[m])
   catEffectsListPC.append(catEffectsPC)
   
   predPC = apply_model.pred_from_effects(modelsP[m]["BASE_VALUE"], len(inputDfC), contEffectsPC, catEffectsPC)
   predsPC.append(predPC)
   overallPredPC += predPC
  
  gradientUC = -gradUC(np.array(inputDfC[target]), overallPredUC, overallPredPC) #d(Loss)/d(predU)
  gradientPC = -gradPC(np.array(inputDfC[target]), overallPredUC, overallPredPC) #d(Loss)/d(predP)
  
  for m in range(len(modelsU)):
   
   model=modelsU[m]
   predUC=predsUC[m]
   
   print("adjust conts")
   
   for col in model["conts"]:
    
    effectOfCol = contEffectsListUC[m][col]
    
    peoc = predUC/effectOfCol #d(predU)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradientUC),contWReleDictListUC[m][col]) #d(Loss)/d(pt) = d(Loss)/d(predU) * d(predU)/d(featpred) * d(featpred)/d(pt)
    
    for k in range(len(finalGradients)):
     totRele = totWReleDictListU[m]["conts"][col][k] + totWReleDictListUC[m]["conts"][col][k]
     if totRele>0:
      modelsU[m]["conts"][col][k][1] -= finalGradients[k]*lrsU[m]/totRele #and not /(sw + swC)
      
   print("adjust cats")
   
   for col in model["cats"]:
    
    effectOfCol = catEffectsListUC[m][col]
    
    peoc = predUC/effectOfCol #d(predU)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradientUC),catWReleDictListUC[m][col]) #d(Loss)/d(pt) = d(Loss)/d(predU) * d(predU)/d(featpred) * d(featpred)/d(pt)
    
    skeys = apply_model.get_sorted_keys(model, col)
    
    #all the uniques . . .
    for k in range(len(skeys)):
     totRele = totWReleDictListU[m]["cats"][col][k] + totWReleDictListUC[m]["cats"][col][k]
     if totRele>0:
      modelsU[m]["cats"][col]["uniques"][skeys[k]] -= finalGradients[k]*lrsU[m]/totRele #and not /(sw+swC)
    
    # . . . and "OTHER"
    totRele = totWReleDictListU[m]["cats"][col][-1] + totWReleDictListUC[m]["cats"][col][-1]
    if totRele>0:
     modelsU[m]["cats"][col]["OTHER"] -= finalGradients[-1]*lrsU[m]/totRele #and not /(sw+swC)
 
  for m in range(len(modelsP)):
   
   model=modelsP[m]
   predPC=predsPC[m]
   
   print("adjust conts")
   
   for col in model["conts"]:
    
    effectOfCol = contEffectsListPC[m][col]
    
    peoc = predPC/effectOfCol #d(predP)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradientPC),contWReleDictListPC[m][col]) #d(Loss)/d(pt) = d(Loss)/d(predP) * d(predP)/d(featpred) * d(featpred)/d(pt)
    
    for k in range(len(finalGradients)):
     totRele = totWReleDictListP[m]["conts"][col][k] + totWReleDictListPC[m]["conts"][col][k]
     if totRele>0:
      modelsP[m]["conts"][col][k][1] -= finalGradients[k]*lrsP[m]/totRele #and not /(sw+swC)
      
   print("adjust cats")
   
   for col in model["cats"]:
    
    effectOfCol = catEffectsListPC[m][col]
    
    peoc = predPC/effectOfCol #d(predP)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradientPC),catWReleDictListPC[m][col]) #d(Loss)/d(pt) = d(Loss)/d(predP) * d(predP)/d(featpred) * d(featpred)/d(pt)
    
    skeys = apply_model.get_sorted_keys(model, col)
    
    #all the uniques . . .
    for k in range(len(skeys)):
     totRele = totWReleDictListP[m]["cats"][col][k] + totWReleDictListPC[m]["cats"][col][k]
     if totRele>0:
      modelsP[m]["cats"][col]["uniques"][skeys[k]] -= finalGradients[k]*lrsP[m]/totRele #and not /(sw+swC)
    
    # . . . and "OTHER"
    totRele = totWReleDictListP[m]["cats"][col][-1] + totWReleDictListPC[m]["cats"][col][-1]
    if totRele>0:
     modelsP[m]["cats"][col]["OTHER"] -= finalGradients[-1]*lrsP[m]/totRele #and not /(sw+swC)
  
 return modelsU, modelsP

if __name__ == '__main__':
 df = pd.read_csv('gnormal.csv')
 modelsU = [{"BASE_VALUE":1.0,"conts":{"x":[[0,103], [7,103]]}, "cats":[]}]
 modelsP = [{"BASE_VALUE":1.0,"conts":{"x":[[0,0.2], [7,0.2]]}, "cats":[]}]
 cdf = df[df['censored']].reset_index()
 udf = df[~df['censored']].reset_index()
 newModelsU, newModelsP = train_model(udf, cdf, "y", 1000, [10], [0.005], modelsU, modelsP)
 for newModel in newModelsU:
  apply_model.explain(newModel)
 for newModel in newModelsP:
  apply_model.explain(newModel)
 

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

def train_model(inputDfs, target, nrounds, lrses, startingModelses, weightses=None, gradses=[[calculus.gnormal_u_diff, calculus.gnormal_p_diff], [calculus.u_diff_censored, calculus.p_diff_censored]]):
 
 modelses = copy.deepcopy(startingModelses)
 
 ws = []
 sws = []
 
 contReleDictListListList = []
 catReleDictListListList = []
 totReleDictListListList = []
 
 contWReleDictListListList = []
 catWReleDictListListList = []
 totWReleDictListListList = []
 
 print("initial weights and relevances setup")
 
 for inputDf in inputDfs:
  
  if weightses==None:
   weights = np.ones(len(inputDf))
  w = np.array(np.transpose(np.matrix(weights)))
  sw = sum(weights)
  
  ws.append(w)
  sws.append(sw)
  
  contReleDictListList = []
  catReleDictListList = []
  totReleDictListList = []
  
  contWReleDictListList = []
  catWReleDictListList = []
  totWReleDictListList = []
  for models in modelses:
   contReleDictList = []
   catReleDictList = []
   totReleDictList = []
   
   contWReleDictList = []
   catWReleDictList = []
   totWReleDictList = []
   for model in models:
    cord = produce_cont_relevances_dict(inputDf,model)
    card = produce_cat_relevances_dict(inputDf,model)
    
    contReleDictList.append(cord)
    catReleDictList.append(card)
    totReleDictList.append(produce_total_relevances_dict(cord, card))
    
    cowrd = produce_wReleDict(cord, w)
    cawrd = produce_wReleDict(card, w)
  
    contWReleDictList.append(cowrd)
    catWReleDictList.append(cawrd)
    totWReleDictList.append(produce_total_relevances_dict(cowrd, cawrd))
   
   contReleDictListList.append(contReleDictList)
   catReleDictListList.append(catReleDictList)
   totReleDictListList.append(totReleDictList)
   
   contWReleDictListList.append(contWReleDictList)
   catWReleDictListList.append(catWReleDictList)
   totWReleDictListList.append(totWReleDictList)
  
  contReleDictListListList.append(contReleDictListList)
  catReleDictListListList.append(catReleDictListList)
  totReleDictListListList.append(totReleDictListList)
  
  contWReleDictListListList.append(contWReleDictListList)
  catWReleDictListListList.append(catWReleDictListList)
  totWReleDictListListList.append(totWReleDictListList)
 
 for i in range(nrounds):
  
  print("epoch: "+str(i+1)+"/"+str(nrounds))
  
  oModelses=copy.deepcopy(modelses)
  
  for d in range(len(inputDfs)):
   inputDf = inputDfs[d]
   
   print("initial pred and effect-gathering")
   
   predses=[]
   overallPreds=[]
   
   contEffectsListList = []
   catEffectsListList = []
   
   gradients = []
   
   for ms in range(len(modelses)):
    models = oModelses[ms]
    
    for model in models:
     apply_model.explain(model)
    
    preds=[]
    overallPred=pd.Series([0]*len(inputDf))
    contEffectsList=[]
    catEffectsList=[]
    
    for m in range(len(models)):
     
     contEffects = apply_model.get_effects_of_cont_cols_from_relevance_dict(contReleDictListListList[d][ms][m],models[m])
     contEffectsList.append(contEffects)
     catEffects = apply_model.get_effects_of_cat_cols_from_relevance_dict(catReleDictListListList[d][ms][m],models[m])
     catEffectsList.append(catEffects)
     
     pred = apply_model.pred_from_effects(models[m]["BASE_VALUE"], len(inputDf), contEffects, catEffects)
     preds.append(pred)
     overallPred += pred
    
    predses.append(preds)
    overallPreds.append(overallPred)
    
    catEffectsListList.append(catEffectsList)
    contEffectsListList.append(contEffectsList)
   
   for ms in range(len(modelses)):
    
    models=oModelses[ms]
    gradient = gradses[d][ms](np.array(inputDf[target]), *overallPreds) #d(Loss)/d(pred)
    
    for m in range(len(models)):
     
     model=models[m]
     pred=predses[ms][m]
     
     print("adjust conts")
     
     for col in model["conts"]:
      
      effectOfCol = contEffectsListList[ms][m][col]
      
      peoc = pred/effectOfCol #d(pred)/d(featpred)
      
      finalGradient = np.matmul(np.array(peoc*gradient),contWReleDictListListList[d][ms][m][col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
      
      for k in range(len(finalGradient)):
       totRele = 0
       for D in range(len(inputDfs)):
        totRele += totWReleDictListListList[D][ms][m]["conts"][col][k]
       if totRele>0:
        modelses[ms][m]["conts"][col][k][1] += finalGradient[k]*lrses[ms][m]/totRele #and not /sum(sws)
      
     print("adjust cats")
     
     for col in model["cats"]:
      
      effectOfCol = catEffectsListList[ms][m][col]
      
      peoc = pred/effectOfCol #d(pred)/d(featpred)
      
      finalGradient = np.matmul(np.array(peoc*gradient),catWReleDictListListList[d][ms][m][col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
      
      skeys = apply_model.get_sorted_keys(model, col)
    
      #all the uniques . . .
      for k in range(len(skeys)):
       totRele = 0
       for D in range(len(inputDfs)):
        totRele = totWReleDictListListList[D][ms][m]["cats"][col][k]
       if totRele>0:
        modelses[ms][m]["cats"][col]["uniques"][skeys[k]] += finalGradient[k]*lrses[ms][m]/totRele #and not /(sw+swC)
      
      # . . . and "OTHER"
      totRele = 0
      for D in range(len(inputDfs)):
       totRele += totWReleDictListListList[D][ms][m]["cats"][col][-1]
      if totRele>0:
       modelses[ms][m]["cats"][col]["OTHER"] += finalGradient[-1]*lrses[ms][m]/totRele #and not /(sw+swC)
  
 return modelses

if __name__ == '__main__':
 df = pd.read_csv('gnormal.csv')
 modelsU = [{"BASE_VALUE":1.0,"conts":{"x":[[0,103], [7,103]]}, "cats":[]}]
 modelsP = [{"BASE_VALUE":1.0,"conts":{"x":[[0,0.2], [7,0.2]]}, "cats":[]}]
 cdf = df[df['censored']].reset_index()
 udf = df[~df['censored']].reset_index()
 newModelses = train_model([udf, cdf], "y", 1000, [[10],[0.005]], [modelsU, modelsP])
 for newModel in newModelses[0]:
  apply_model.explain(newModel)
 for newModel in newModelses[1]:
  apply_model.explain(newModel)
 

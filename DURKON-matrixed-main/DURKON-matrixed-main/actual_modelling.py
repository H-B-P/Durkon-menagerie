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

def train_model(inputDf, target, nrounds, lrs, startingModels, weights=None, grad=calculus.Poisson_grad):
 
 models = copy.deepcopy(startingModels)
 
 if weights==None:
  weights = np.ones(len(inputDf))
 w = np.array(np.transpose(np.matrix(weights)))
 sw = sum(weights)
 
 contReleDictList=[]
 catReleDictList=[]
 totReleDictList=[]
 
 contWReleDictList=[]
 catWReleDictList=[]
 totWReleDictList=[]
 
 print("initial relevances setup")
 
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
 
 for i in range(nrounds):
  
  print("epoch: "+str(i+1)+"/"+str(nrounds))
  for model in models:
   apply_model.explain(model)
  
  print("initial pred and effect-gathering")
  
  preds=[]
  overallPred=pd.Series([0]*len(inputDf))
  contEffectsList=[]
  catEffectsList=[]
  
  for m in range(len(models)):
   
   contEffects = apply_model.get_effects_of_cont_cols_from_relevance_dict(contReleDictList[m],models[m])
   contEffectsList.append(contEffects)
   catEffects = apply_model.get_effects_of_cat_cols_from_relevance_dict(catReleDictList[m],models[m])
   catEffectsList.append(catEffects)
   
   pred = apply_model.pred_from_effects(models[m]["BASE_VALUE"], len(inputDf), contEffects, catEffects)
   preds.append(pred)
   overallPred += pred
  
  gradient = grad(overallPred, np.array(inputDf[target])) #d(Loss)/d(pred)
  
  for m in range(len(models)):
   
   model=models[m]
   pred=preds[m]
   
   print("adjust conts")
   
   for col in model["conts"]:
    
    effectOfCol = contEffectsList[m][col]
    
    peoc = pred/effectOfCol #d(pred)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradient),contWReleDictList[m][col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    
    for k in range(len(finalGradients)):
     totRele = totWReleDictList[m]["conts"][col][k]
     if totRele>0:
      models[m]["conts"][col][k][1] -= finalGradients[k]*lrs[m]/totRele #and not /sw
       
      
   print("adjust cats")
   
   for col in model["cats"]:
    
    effectOfCol = catEffectsList[m][col]
    
    peoc = pred/effectOfCol #d(pred)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradient),catWReleDictList[m][col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    
    skeys = apply_model.get_sorted_keys(model, col)
    
    #all the uniques . . .
    for k in range(len(skeys)):
     totRele = totWReleDictList[m]["cats"][col][k]
     if totRele>0:
      models[m]["cats"][col]["uniques"][skeys[k]] -= finalGradients[k]*lrs[m]/totRele #and not /sw
    
    # . . . and "OTHER"
    totRele = totWReleDictList[m]["cats"][col][-1]
    if totRele>0:
     models[m]["cats"][col]["OTHER"] -= finalGradients[-1]*lrs[m]/totRele #and not /sw
 return models

if __name__ == '__main__':
 df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8,9],"y":[1,2,3,4,5,6,7,8,9]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[1,5],[5,5],[9,5]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [2], models,weights=[1,2,3,4,5,6,7,8,9])
 for newModel in newModels:
  apply_model.explain(newModel)


if False:
 df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8,9],"y":[2,2,2,2,2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[2,1],[5,1],[8,1]]}, "cats":[]}]
 print(produce_cont_relevances(df, models[0], "x"))
 
 df = pd.DataFrame({"x":["cat","dog","cat","mouse","rat"],"y":[2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"x":{"uniques":{"cat":1,"mouse":1,"dog":1},"OTHER":1}}}]
 print(produce_cat_relevances(df, models[0], "x"))
 
 df = pd.DataFrame({"x":[1,2,3,4,5,6,7,8,9],"y":[2,2,2,2,2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[2,1],[5,2],[8,1]]}, "cats":[]}]
 reles = produce_cont_relevances(df, models[0], "x")
 print(apply_model.get_effect_of_this_cont_col_from_relevances(reles, models[0], "x"))
 
 df = pd.DataFrame({"x":["cat","dog","cat","mouse","rat"],"y":[2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"x":{"uniques":{"cat":2,"mouse":1,"dog":3},"OTHER":1.5}}}]
 reles = produce_cat_relevances(df, models[0], "x")
 print(apply_model.get_effect_of_this_cat_col_from_relevances(reles, models[0], "x"))
 
 df = pd.DataFrame({"x":[1,2,3],"y":[2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":[]}]
 newModels = train_model(df, "y",50, [2.0], models)
 for newModel in newModels:
  apply_model.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2,3],"y":[2,3,4]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[0,1],[4,1]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [2.0], models)
 for newModel in newModels:
  apply_model.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2],"y":[2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[1,1],[2,1]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [2.0], models)
 for newModel in newModels:
  apply_model.explain(newModel)
 
 df = pd.DataFrame({"x":[1, 995,996,997,998,999,1000],"y":[2,2,2,2,2,2,2]})
 models = [{"BASE_VALUE":1.0,"conts":{"x":[[1,1],[1000,1]]}, "cats":[]}]
 newModels = train_model(df, "y",50, [1.5], models)
 for newModel in newModels:
  apply_model.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2,3,4],"y":[1+1,2+1.2,3+1.4,4+1.6]})
 models = [{"BASE_VALUE":2,"conts":{"x":[[1,1],[4,1]]}, "cats":[]},{"BASE_VALUE":1,"conts":{"x":[[1,1],[4,1]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [1.5, 1.5], models)
 for newModel in newModels:
  apply_model.explain(newModel)

 df = pd.DataFrame({"x":["cat","dog","cat","mouse","rat"],"y":[2.0,3.0,2.0,1.0,1.5]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"x":{"uniques":{"cat":1,"mouse":1,"dog":1},"OTHER":1}}}]
 newModels = train_model(df, "y",100, [0.5], models)
 for newModel in newModels:
  apply_model.explain(newModel)
 
 df = pd.DataFrame({"x":[1,2,3,4],"y":[1+1,2+1.2,3+1.4,4+1.6]})
 models = [{"BASE_VALUE":2,"conts":{"x":[[1,1],[4,3]]}, "cats":[]},{"BASE_VALUE":1,"conts":{"x":[[1,0.4],[4,0.6]]}, "cats":[]}]
 newModels = train_model(df, "y",100, [0,0], models, 10.0)
 for newModel in newModels:
  apply_model.explain(newModel)
 

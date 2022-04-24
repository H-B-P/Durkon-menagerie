import pandas as pd
import numpy as np
import math
import copy
import time

import util
import apply_model
import calculus

def produce_cont_relevances(inputSeries, cont):
 reles=np.zeros((len(cont)+1,len(inputSeries)))
 
 reles[0][(inputSeries<=cont[0][0])] = 1 #d(featpred)/d(pt)
 for i in range(len(cont)-1):
  x = inputSeries
  x1 = cont[i][0]
  x2 = cont[i+1][0]
  subset = (x>=x1) & (x<=x2)
  #print(reles[subset][:,1])
  reles[i][subset] = (x2 - x[subset])/(x2 - x1) #d(featpred)/d(pt)
  reles[i+1][subset] = (x[subset] - x1)/(x2 - x1) #d(featpred)/d(pt)
 reles[-2][(inputSeries>=cont[-1][0])] = 1 #d(featpred)/d(pt)
 
 reles[-1] = 1 - np.sum(reles, axis=0)
 return np.transpose(reles) #roundabout way of doing this but the rest of the function doesn't flow as naturally if x and y don't switch places

def produce_cont_relevances_dict(inputDf, model):
 opDict = {}
 
 for col in model["conts"]:
  opDict[col]=produce_cont_relevances(inputDf[col], model['conts'][col])
 
 return opDict

def produce_cat_relevances(inputSeries, cat):
 reles=np.zeros((len(cat["uniques"])+1,len(inputSeries)))
 
 skeys = apply_model.get_sorted_keys(cat)
 for i in range(len(skeys)):
  reles[i][inputSeries.isin([skeys[i]])] = 1 #d(featpred)/d(pt)
 reles[-1][~inputSeries.isin(skeys)] = 1 #d(featpred)/d(pt)
 
 return np.transpose(reles) #roundabout way of doing this but the rest of the function doesn't flow as naturally if x and y don't switch places

def produce_cat_relevances_dict(inputDf, model):
 opDict = {}
 
 for col in model["cats"]:
  opDict[col]=produce_cat_relevances(inputDf[col], model['cats'][col])
 
 return opDict

#Interactions

def interact_relevances(relesA, relesB):
 
 relesA = np.transpose(relesA) #Yes, I know.
 relesB = np.transpose(relesB) #Shut up.
 
 relesI = np.zeros((len(relesA)*len(relesB),len(relesA[0])))
 
 for i in range(len(relesA)):
  for j in range(len(relesB)):
   relesI[i*len(relesB)+j] = relesA[i]*relesB[j]
 
 return np.transpose(relesI) # . . .

def produce_catcat_relevances(inputDf, model, cols):
 col1, col2 = cols.split(' X ')
 return interact_relevances(produce_cat_relevances(inputDf[col1], model['catcats'][cols]), produce_cat_relevances(inputDf[col2], model['catcats'][cols]["OTHER"]))

def produce_catcont_relevances(inputDf, model, cols):
 col1, col2 = cols.split(' X ')
 return interact_relevances(produce_cat_relevances(inputDf[col1], model['catconts'][cols]), produce_cont_relevances(inputDf[col2], model['catconts'][cols]["OTHER"]))

def produce_contcont_relevances(inputDf, model, cols):
 col1, col2 = cols.split(' X ')
 return interact_relevances(produce_cont_relevances(inputDf[col1], model['contconts'][cols]), produce_cont_relevances(inputDf[col2], model['contconts'][cols][0][1]))

def produce_interxn_relevances_dict(inputDf, model):
 
 opDict = {}
 
 for cols in model['catcats']:
  opDict[cols] = produce_catcat_relevances(inputDf, model, cols)
 
 for cols in model['catconts']:
  opDict[cols] = produce_catcont_relevances(inputDf, model, cols)
 
 for cols in model['contconts']:
  opDict[cols] = produce_contcont_relevances(inputDf, model, cols)
 
 return opDict

def sum_and_listify_matrix(a):
 return np.array(sum(a)).tolist()

def produce_total_irelevances_dict(releDict):
 op = {}
 for cols in releDict:
  op[cols] = sum_and_listify_matrix(releDict[cols])
 return op

def produce_total_relevances_dict(contReleDict, catReleDict):
 op = {"conts":{},"cats":{}}
 for col in contReleDict:
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

def train_model(inputDf, target, nrounds, lrs, startingModels, weights=None, ignoreCols = [], grad=calculus.Poisson_grad):
 
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
 
 #Interactions . . .
 
 interxReleDictList = []
 interxTotReleDictList = []
 
 interxWReleDictList = []
 interxTotWReleDictList = []
 
 for model in models:
  ird = produce_interxn_relevances_dict(inputDf, model)
  
  interxReleDictList.append(ird)
  interxTotReleDictList.append(produce_total_irelevances_dict(ird))
  
  wird = produce_wReleDict(ird, w)
  
  interxWReleDictList.append(wird)
  interxTotWReleDictList.append(produce_total_irelevances_dict(wird))
 
 for i in range(nrounds):
  
  print("epoch: "+str(i+1)+"/"+str(nrounds))
  for model in models:
   print(model)#apply_model.explain(model)
  
  print("initial pred and effect-gathering")
  
  preds=[]
  overallPred=pd.Series([0]*len(inputDf))
  contEffectsList=[]
  catEffectsList=[]
  interxEffectsList=[]
  
  for m in range(len(models)):
   
   contEffects = apply_model.get_effects_of_cont_cols_from_relevance_dict(contReleDictList[m],models[m])
   contEffectsList.append(contEffects)
   catEffects = apply_model.get_effects_of_cat_cols_from_relevance_dict(catReleDictList[m],models[m])
   catEffectsList.append(catEffects)
   
   interxEffects = apply_model.get_effects_of_interxns_from_relevance_dict(interxReleDictList[m],models[m])
   interxEffectsList.append(interxEffects)
   
   pred = apply_model.pred_from_effects(models[m]["BASE_VALUE"], len(inputDf), contEffects, catEffects, interxEffects)
   preds.append(pred)
   overallPred += pred
  
  gradient = grad(overallPred, np.array(inputDf[target])) #d(Loss)/d(pred)
  
  for m in range(len(models)):
   
   model=models[m]
   pred=preds[m]
   
   print("adjust conts")
   
   for col in [c for c in model['conts'] if c not in ignoreCols]:#model["conts"]:
    
    effectOfCol = contEffectsList[m][col]
    
    peoc = pred/effectOfCol #d(pred)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradient),contWReleDictList[m][col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    
    for k in range(len(models[m]['conts'][col])):
     totRele = totWReleDictList[m]["conts"][col][k]
     if totRele>0:
      models[m]["conts"][col][k][1] -= finalGradients[k]*lrs[m]/totRele #and not /sw
       
      
   print("adjust cats")
   
   for col in [c for c in model['cats'] if c not in ignoreCols]:#model["cats"]:
    
    effectOfCol = catEffectsList[m][col]
    
    peoc = pred/effectOfCol #d(pred)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradient),catWReleDictList[m][col]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    
    skeys = apply_model.get_sorted_keys(model['cats'][col])
    
    #all the uniques . . .
    for k in range(len(skeys)):
     totRele = totWReleDictList[m]["cats"][col][k]
     if totRele>0:
      models[m]["cats"][col]["uniques"][skeys[k]] -= finalGradients[k]*lrs[m]/totRele #and not /sw
    
    # . . . and "OTHER"
    totRele = totWReleDictList[m]["cats"][col][-1]
    if totRele>0:
     models[m]["cats"][col]["OTHER"] -= finalGradients[-1]*lrs[m]/totRele #and not /sw
   
   print('adjust catcats')
   
   for cols in [c for c in model['catcats'] if c not in ignoreCols]:#model['catcats']:
    
    effectOfCols = interxEffectsList[m][cols]
    
    peoc = pred/effectOfCols #d(pred)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradient),interxWReleDictList[m][cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    
    skeys1 = apply_model.get_sorted_keys(model['catcats'][cols])
    skeys2 = apply_model.get_sorted_keys(model['catcats'][cols]["OTHER"])
    
    for i in range(len(skeys1)):
     for j in range(len(skeys2)):
      totRele = interxTotWReleDictList[m][cols][i*(len(skeys2)+1)+j]
      if totRele>0:
       models[m]["catcats"][cols]["uniques"][skeys1[i]]['uniques'][skeys2[i]] -= finalGradients[i*(len(skeys2)+1)+j]*lrs[m]/totRele #and not /sw
     totRele = interxTotWReleDictList[m][cols][i*(len(skeys2)+1)+len(skeys2)]
     if totRele>0:
      models[m]['catcats'][cols]["uniques"][skeys1[i]]['OTHER'] -= finalGradients[i*(len(skeys2)+1)+len(skeys2)]*lrs[m]/totRele #and not /sw
    
    for j in range(len(skeys2)):
     totRele = interxTotWReleDictList[m][cols][len(skeys1)*(len(skeys2)+1)+j]
     if totRele>0:
      models[m]["catcats"][cols]['OTHER']['uniques'][skeys2[i]] -= finalGradients[len(skeys1)*(len(skeys2)+1)+j]*lrs[m]/totRele #and not /sw
     
    totRele = interxTotWReleDictList[m][cols][-1]
    if totRele>0:
     models[m]['catcats'][cols]['OTHER']['OTHER'] -= finalGradients[-1]*lrs[m]/totRele #and not /sw
   
   print('adjust catconts')
   
   for cols in [c for c in model['catconts'] if c not in ignoreCols]:#model['catconts']
    
    effectOfCols = interxEffectsList[m][cols]
    
    peoc = pred/effectOfCols #d(pred)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradient),interxWReleDictList[m][cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    
    skeys = apply_model.get_sorted_keys(model['catconts'][cols])
    
    for i in range(len(skeys)):
     for j in range(len(model['catconts'][cols]["OTHER"])):
      totRele = interxTotWReleDictList[m][cols][i*(len(model['catconts'][cols]["OTHER"])+1)+j]
      if totRele>0:
       models[m]['catconts'][cols]['uniques'][skeys[i]][j][1] -= finalGradients[i*(len(model['catconts'][cols]["OTHER"])+1)+j]*lrs[m]/totRele #and not /sw
    
    for j in range(len(model['catconts'][cols]["OTHER"])):
     totRele = interxTotWReleDictList[m][cols][len(skeys)*(len(model['catconts'][cols]["OTHER"])+1)+j]
     if totRele>0:
      models[m]['catconts'][cols]['OTHER'][j][1] -= finalGradients[len(skeys)*(len(model['catconts'][cols]["OTHER"])+1)+j]*lrs[m]/totRele #and not /sw
   
   for cols in [c for c in model['contconts'] if c not in ignoreCols]:#model['contconts']
    
    effectOfCols = interxEffectsList[m][cols]
    
    peoc = pred/effectOfCols #d(pred)/d(featpred)
    
    finalGradients = np.matmul(np.array(peoc*gradient),interxWReleDictList[m][cols]) #d(Loss)/d(pt) = d(Loss)/d(pred) * d(pred)/d(featpred) * d(featpred)/d(pt)
    
    for i in range(len(model['contconts'][cols])):
     for j in range(len(model['contconts'][cols][0][1])):
      totRele = interxTotWReleDictList[m][cols][i*(len(model['contconts'][cols][0][1])+1)+j]
      if totRele>0:
       models[m]['contconts'][cols][i][1][j][1] -= finalGradients[i*(len(model['contconts'][cols][0][1])+1)+j]*lrs[m]/totRele #and not /sw
    
 return models

if __name__ == '__main__':
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q'],'cat2':['c','d','q','c','d','q','c','d','q','q'],"y":[1,2,3,4,5,6,7,8,9,10]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"cat1":{"uniques":{"a":1,"b":1,},"OTHER":1}, "cat2":{"uniques":{"c":1,"d":1},"OTHER":1}}, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'c':1,'d':1},'OTHER':1},'b':{'uniques':{'c':1,'d':1},'OTHER':1}},'OTHER':{'uniques':{'c':1,'d':1},'OTHER':1}}}, 'catconts':{}, 'contconts':{}}]
 print(produce_catcat_relevances(df, models[0], "cat1 X cat2"))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q'],'cat2':['c','d','q','c','d','q','c','d','q','q'],"y":[1,2,3,4,5,6,7,8,9,10]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"cat1":{"uniques":{"a":1,"b":1,},"OTHER":1}, "cat2":{"uniques":{"c":1,"d":1},"OTHER":1}}, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'c':1,'d':1},'OTHER':1},'b':{'uniques':{'c':1,'d':2},'OTHER':1}},'OTHER':{'uniques':{'c':1,'d':1},'OTHER':1}}}, 'catconts':{}, 'contconts':{}}]
 reles = produce_catcat_relevances(df, models[0], "cat1 X cat2")
 print(apply_model.get_effect_of_this_catcat_from_relevances(reles, models[0], "cat1 X cat2"))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q'],'cat2':['c','d','q','c','d','q','c','d','q','q'],"y":[1,1,1,1,2,1,1,1,1,1]})
 models = [{"BASE_VALUE":1.0,"conts":{}, "cats":{"cat1":{"uniques":{"a":1,"b":1,},"OTHER":1}, "cat2":{"uniques":{"c":1,"d":1},"OTHER":1}}, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'c':1,'d':1},'OTHER':1},'b':{'uniques':{'c':1,'d':1},'OTHER':1}},'OTHER':{'uniques':{'c':1,'d':1},'OTHER':1}}}, 'catconts':{}, 'contconts':{}}]
 
 newModels = train_model(df, "y",50, [0.4], models, ignoreCols = ['cat1','cat2'])
 print(newModels)
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 models = [{"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]]}, "cats":{"cat1":{"uniques":{"a":1,"b":1},"OTHER":1}}, 'catcats':{}, 'catconts':{"cat1 X cont1":{"uniques":{"a":[[1,1],[2,1],[3,1]],"b":[[1,1],[2,1],[3,1]]},"OTHER":[[1,1],[2,1],[3,1]]}}, 'contconts':{}}]
 print(produce_catcont_relevances(df, models[0], "cat1 X cont1"))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 models = [{"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]]}, "cats":{"cat1":{"uniques":{"a":1,"b":1},"OTHER":1}}, 'catcats':{}, 'catconts':{"cat1 X cont1":{"uniques":{"a":[[1,1],[2,1],[3,1]],"b":[[1,1],[2,4],[3,1]]},"OTHER":[[1,1],[2,1],[3,1]]}}, 'contconts':{}}]
 reles = produce_catcont_relevances(df, models[0], "cat1 X cont1")
 print(apply_model.get_effect_of_this_catcont_from_relevances(reles, models[0], "cat1 X cont1"))
 
 df = pd.DataFrame({"cat1":['a','a','a','b','b','b','q','q','q','q','q'],'cont1':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,1,1,1,1,2,1,1,1,1,1]})
 models = [{"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]]}, "cats":{"cat1":{"uniques":{"a":1,"b":1},"OTHER":1}}, 'catcats':{}, 'catconts':{"cat1 X cont1":{"uniques":{"a":[[1,1],[2,1],[3,1]],"b":[[1,1],[2,1],[3,1]]},"OTHER":[[1,1],[2,1],[3,1]]}}, 'contconts':{}}]
 newModels = train_model(df, "y",50, [0.4], models, ignoreCols = ['cat1','cont1'])
 print(newModels)
 
 df = pd.DataFrame({"cont1":[1,1,1,2,2,2,3,3,3,1.5,np.nan],'cont2':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 models = [{"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]],'cont2':[[1,1],[2,1],[3,1]]}, "cats":{}, 'catcats':{}, 'catconts':{}, 'contconts':{'cont1 X cont2': [[1,[[1,1],[2,1],[3,1]]],[2,[[1,1],[2,1],[3,1]]],[3,[[1,1],[2,1],[3,1]]]]} }]
 print(produce_contcont_relevances(df, models[0], "cont1 X cont2"))
 
 df = pd.DataFrame({"cont1":[1,1,1,2,2,2,3,3,3,1.5,np.nan],'cont2':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,2,3,4,5,6,7,8,9,10,11]})
 models = [{"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]],'cont2':[[1,1],[2,1],[3,1]]}, "cats":{}, 'catcats':{}, 'catconts':{}, 'contconts':{'cont1 X cont2': [[1,[[1,1],[2,1],[3,1]]],[2,[[1,1],[2,1],[3,1]]],[3,[[1,1],[2,1],[3,5]]]]} }]
 reles = produce_contcont_relevances(df, models[0], "cont1 X cont2")
 print(apply_model.get_effect_of_this_contcont_from_relevances(reles, models[0], "cont1 X cont2"))
 
 df = pd.DataFrame({"cont1":[1,1,1,2,2,2,3,3,3,1.5,np.nan],'cont2':[1,2,3,1,2,3,1,2,3,1.5,np.nan],"y":[1,1,1, 1,1,1, 1,1,2, 1,1]})
 models = [{"BASE_VALUE":1.0,"conts":{'cont1':[[1,1],[2,1],[3,1]],'cont2':[[1,1],[2,1],[3,1]]}, "cats":{}, 'catcats':{}, 'catconts':{}, 'contconts':{'cont1 X cont2': [[1,[[1,1],[2,1],[3,1]]],[2,[[1,1],[2,1],[3,1]]],[3,[[1,1],[2,1],[3,1]]]]} }]
 newModels = train_model(df, "y",50, [0.4], models, ignoreCols = ['cont1','cont2'])
 print(newModels)
 
if False:
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
 

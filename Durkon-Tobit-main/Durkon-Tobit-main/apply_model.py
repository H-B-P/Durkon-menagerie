import pandas as pd
import numpy as np
import math
import copy
import time

import util

def de_feat(model, boringValue=1):
 oldModel=copy.deepcopy(model)
 newModel={"BASE_VALUE":oldModel["BASE_VALUE"], "conts":{}, "cats":{}}
 
 for col in oldModel["conts"]:
  empty=True
  for pt in oldModel["conts"][col]:
   if pt[1]!=boringValue:
    empty=False
  if not empty:
   newModel["conts"][col]=oldModel["conts"][col]
 
 for col in oldModel["cats"]:
  empty=True
  if oldModel["cats"][col]["OTHER"]!=boringValue:
   empty=False
  for unique in oldModel["cats"][col]["uniques"]:
   if oldModel["cats"][col]["uniques"][unique]!=boringValue:
    empty=False
  if not empty:
   newModel["cats"][col]=oldModel["cats"][col]
 
 return newModel

def get_sorted_keys(model, col):
 keys = [c for c in model["cats"][col]["uniques"]]
 keys.sort()
 return keys

def get_effect_of_this_cont_col_from_relevances(reles, model, col):
 postmultmat = np.array([pt[1] for pt in model["conts"][col]])
 return np.matmul(reles,postmultmat)

def get_effects_of_cont_cols_from_relevance_dict(releDict, model):
 opDict = {}
 for col in model["conts"]:
  opDict[col]= get_effect_of_this_cont_col_from_relevances(releDict[col], model, col)
 return opDict
 
def get_effect_of_this_cat_col_from_relevances(reles, model, col):
 skeys = get_sorted_keys(model,col)
 postmultmat = np.array([model["cats"][col]["uniques"][key] for key in skeys]+[model["cats"][col]["OTHER"]])
 return np.matmul(reles,postmultmat)
 
def get_effects_of_cat_cols_from_relevance_dict(releDict, model):
 opDict = {}
 for col in model["cats"]:
  opDict[col]= get_effect_of_this_cat_col_from_relevances(releDict[col], model, col)
 return opDict

def pred_from_effects(base,l,contEffs,catEffs):
 op = pd.Series([base]*l)
 for col in contEffs:
  op = op*contEffs[col]
 for col in catEffs:
  op = op*catEffs[col]
 return op

def predict(inputDf, model):
 preds = pd.Series([model["BASE_VALUE"]]*len(inputDf))
 for col in model["conts"]:
  effectOfCol = get_effect_of_this_cont_col(inputDf, model, col)
  preds = preds*effectOfCol
 for col in model["cats"]:
  effectOfCol = get_effect_of_this_cat_col(inputDf, model, col)
  preds = preds*effectOfCol
 return preds

def get_effect_of_this_cont_col(inputDf, model, col):
 x = inputDf[col]
 effectOfCol = pd.Series([1]*len(inputDf))
 effectOfCol.loc[(x<=model["conts"][col][0][0])] = model["conts"][col][0][1] #Everything too early gets with the program
 for i in range(len(model["conts"][col])-1):
  x1 = model["conts"][col][i][0]
  x2 = model["conts"][col][i+1][0]
  y1 = model["conts"][col][i][1]
  y2 = model["conts"][col][i+1][1]
  effectOfCol.loc[(x>=x1)&(x<=x2)] = ((x-x1)*y2 + (x2-x)*y1)/(x2 - x1)
 effectOfCol.loc[x>=model["conts"][col][-1][0]] = model["conts"][col][-1][1] #Everything too late gets with the program
 return effectOfCol

def get_effect_of_this_cont_col_on_single_input(x, model, col):
 if x<=model["conts"][col][0][0]:
  return model["conts"][col][0][1] #everything outside our scope is flat, we ignore the details.
 for i in range(len(model["conts"][col])-1):
  if (x>=model["conts"][col][i][0] and x<=model["conts"][col][i+1][0]):
   return ((x-model["conts"][col][i][0])*model["conts"][col][i+1][1] + (model["conts"][col][i+1][0]-x)*model["conts"][col][i][1])/(model["conts"][col][i+1][0]-model["conts"][col][i][0])#((x-p1)y1 + (p2-x)y2) / (p2 - p1)
 if x>=model["conts"][col][len(model["conts"][col])-1][0]:
  return model["conts"][col][len(model["conts"][col])-1][1]
 return "idk lol"

def get_effect_of_this_cat_col_on_single_input(x,model,col): #slightly roundabout approach so we can copy for columns
 for unique in model["cats"][col]["uniques"]:
  if x==unique:
   return model["cats"][col]["uniques"][unique]
 return model["cats"][col]["OTHER"]

def get_effect_of_this_cat_col(inputDf, model, col):
 effectOfCol = pd.Series([model["cats"][col]["OTHER"]]*len(inputDf))
 for unique in model["cats"][col]["uniques"]:
  effectOfCol[inputDf[col]==unique] = model["cats"][col]["uniques"][unique]
 return effectOfCol

def round_to_sf(x, sf=5):
 if x==0:
  return 0
 else:
  return round(x,sf-1-int(math.floor(math.log10(abs(x)))))

def roundify_dict(dyct, sf=5):
 opdyct=dyct.copy()
 for k in opdyct:
  if k=="uniques":
   for unique in opdyct[k]:
    opdyct[k][unique] = round(opdyct[k][unique], sf)#round_to_sf(opdyct[k][unique], sf)
  else:
   opdyct[k]=round(opdyct[k], sf)
 return opdyct

def roundify_ptlist(ptlyst, sf=5):
 oplyst = copy.deepcopy(ptlyst)
 for i in range(len(oplyst)):
  oplyst[i][1] = round(oplyst[i][1],sf)
 return oplyst

def explain(model, sf=5):
 print("BASE_VALUE", round_to_sf(model["BASE_VALUE"], sf))
 for col in model["conts"]:
  print(col, roundify_ptlist(model["conts"][col], sf))
 for col in model["cats"]:
  print(col, roundify_dict(model["cats"][col], sf))
 print("-")

def prep_starting_model(inputDf, conts, pts, cats, uniques, target, boringValue=1, frac=1):
 
 model={"BASE_VALUE":inputDf[target].mean()*frac, "conts":{}, "cats":{}}
 
 for col in conts:
  model["conts"][col]=[]
  for pt in pts[col]:
   model["conts"][col].append([pt,boringValue])
 
 for col in cats:
  model["cats"][col]={"OTHER":boringValue}
  model["cats"][col]["uniques"]={}
  for unique in uniques[col]:
   model["cats"][col]["uniques"][unique]=boringValue
 
 return model

def normalize_model(model, totReleDict):
 
 opModel = copy.deepcopy(model)
 
 for col in totReleDict["conts"]:
  relaTimesRele = 0
  for i in range(len(opModel["conts"][col])):
   relaTimesRele += opModel["conts"][col][i][1] * totReleDict["conts"][col][i]
  averageRela = relaTimesRele/sum(totReleDict["conts"][col])
  for i in range(len(opModel["conts"][col])):
   opModel["conts"][col][i][1] /= averageRela
  opModel["BASE_VALUE"] *= averageRela
 
 for col in totReleDict["cats"]:
  relaTimesRele = 0
  skeys = get_sorted_keys(model, col)
  for i in range(len(skeys)):
   relaTimesRele += opModel["cats"][col]["uniques"][skeys[i]] * totReleDict["cats"][col][i]
  relaTimesRele += opModel["cats"][col]["OTHER"] * totReleDict["cats"][col][-1]
  averageRela = relaTimesRele/sum(totReleDict["cats"][col])
  for i in range(len(skeys)):
   opModel["cats"][col]["uniques"][skeys[i]] /= averageRela
  opModel["cats"][col]["OTHER"] /= averageRela
  opModel["BASE_VALUE"] *= averageRela
 
 return opModel

def enforce_min_rela(model, minRela=0.1): #I could generalize this to apply an arbitrary function
 
 opModel = copy.deepcopy(model)
 
 for col in opModel["conts"]:
  for i in range(len(opModel["conts"][col])):
   opModel["conts"][col][i][1] = max(minRela, opModel["conts"][col][i][1])
 
 for col in opModel["cats"]:
  for u in opModel["cats"][col]["uniques"]:
   opModel["cats"][col]["uniques"][u] = max(minRela, opModel["cats"][col]["uniques"][u])
  opModel["cats"][col]["OTHER"] = max(minRela, opModel["cats"][col]["OTHER"])
 
 return opModel


def caricature_this_cont_col(model, col, mult=1,frac=1,boringValue=1):
 
 opModel = copy.deepcopy(model)
 
 opModel["BASE_VALUE"] *= frac
 
 for i in range(len(opModel["conts"][col])):
  opModel["conts"][col][i][1] = boringValue + mult*(opModel["conts"][col][i][1]-boringValue)
 
 return opModel


def caricature_this_cat_col(model, col, mult=1,frac=1,boringValue=1):
 
 opModel = copy.deepcopy(model)
 
 opModel["BASE_VALUE"] *= frac
 
 for u in opModel["cats"][col]["uniques"]:
  opModel["cats"][col]["uniques"][u] = boringValue + mult*(opModel["cats"][col]["uniques"][u]-boringValue)
 
 opModel["cats"][col]["OTHER"] = boringValue + mult*(opModel["cats"][col]["OTHER"]-boringValue)
 
 return opModel


def caricature_model(model, mult=1, frac=0.5, boringValue=1):
 
 opModel = copy.deepcopy(model)
 
 opModel["BASE_VALUE"] *= frac
 
 for col in opModel["conts"]:
  opModel = caricature_this_cont_col(opModel, col, mult, 1, boringValue)
 
 for col in opModel["cats"]:
  opModel = caricature_this_cat_col(opModel, col, mult, 1, boringValue)
 
 return opModel


if __name__ == '__main__':
 exampleModel = {"BASE_VALUE":1700,"conts":{"cont1":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "cont2":[[37,1.2],[98, 0.9]]}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "florpalorp":0.92}, "OTHER":1.04}}}
 exampleDf = pd.DataFrame({"cont1":[0.013,0.015,0.025, 0.035], "cont2":[37,48,45,51], "cat1":["wstfgl","florpalorp","dukis","welp"], "y":[5,7,9,11]})
 
 print(get_effect_of_this_cont_col_on_single_input(0.012, exampleModel, "cont1")) #should be 1.02
 print(get_effect_of_this_cont_col_on_single_input(0.04, exampleModel, "cont1")) #should be 1.06
 print(get_effect_of_this_cat_col_on_single_input("florpalorp", exampleModel, "cat1")) #should be 0.92
 print(get_effect_of_this_cat_col_on_single_input(12, exampleModel, "cat1")) #should be 1.04
 
 print(list(get_effect_of_this_cat_col(exampleDf, exampleModel, "cat1"))) #[1.05,0.92,1.04,1.04]
 print(list(get_effect_of_this_cont_col(exampleDf, exampleModel, "cont1"))) #[1.03,1.05,1.08,1.06]

 print(caricature_model(exampleModel,2, 0.5))
 
 

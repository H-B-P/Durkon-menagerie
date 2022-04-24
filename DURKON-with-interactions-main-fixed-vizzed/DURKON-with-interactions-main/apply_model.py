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

def get_sorted_keys(cat):
 keys = [c for c in cat["uniques"]]
 keys.sort()
 return keys

def get_effect_of_this_cont_col_from_relevances(reles, model, col, boringValue=1):
 postmultmat = np.array([pt[1] for pt in model["conts"][col]]+[boringValue])
 return np.matmul(reles,postmultmat)

def get_effects_of_cont_cols_from_relevance_dict(releDict, model):
 opDict = {}
 for col in model["conts"]:
  opDict[col]= get_effect_of_this_cont_col_from_relevances(releDict[col], model, col)
 return opDict
 
def get_effect_of_this_cat_col_from_relevances(reles, model, col):
 skeys = get_sorted_keys(model['cats'][col])
 postmultmat = np.array([model["cats"][col]["uniques"][key] for key in skeys]+[model["cats"][col]["OTHER"]])
 return np.matmul(reles,postmultmat)
 
def get_effects_of_cat_cols_from_relevance_dict(releDict, model):
 opDict = {}
 for col in model["cats"]:
  opDict[col]= get_effect_of_this_cat_col_from_relevances(releDict[col], model, col)
 return opDict

def get_effect_of_this_catcat_from_relevances(reles, model, cols):
 skeys1 = get_sorted_keys(model['catcats'][cols])
 skeys2 = get_sorted_keys(model['catcats'][cols]["OTHER"])
 postmultmat = []
 for key1 in skeys1:
  postmultmat = postmultmat+[model['catcats'][cols]['uniques'][key1]['uniques'][key2] for key2 in skeys2]+[model['catcats'][cols]['uniques'][key1]['OTHER']]
 postmultmat = postmultmat + [model['catcats'][cols]['OTHER']['uniques'][key2] for key2 in skeys2]+ [model['catcats'][cols]['OTHER']['OTHER']]
 postmultmat = np.array(postmultmat)
 return np.matmul(reles, postmultmat)
 
def get_effect_of_this_catcont_from_relevances(reles, model, cols, boringValue=1):
 skeys = get_sorted_keys(model["catconts"][cols])
 postmultmat = []
 for key in skeys:
  postmultmat = postmultmat + [pt[1] for pt in model["catconts"][cols]['uniques'][key]]+[boringValue]
 postmultmat = postmultmat + [pt[1] for pt in model["catconts"][cols]['OTHER']] + [boringValue]
 postmultmat = np.array(postmultmat)
 return np.matmul(reles, postmultmat)
 
def get_effect_of_this_contcont_from_relevances(reles, model, cols, boringValue=1):
 postmultmat = []
 for pt1 in model['contconts'][cols]:
  postmultmat = postmultmat + [pt2[1] for pt2 in pt1[1]] + [boringValue]
 postmultmat = postmultmat + [boringValue]*(len(model['contconts'][cols][-1][1])+1)
 postmultmat = np.array(postmultmat)
 return np.matmul(reles, postmultmat)
 
def get_effects_of_interxns_from_relevance_dict(releDict, model):
 opDict = {}
 for cols in model['catcats']:
  opDict[cols] = get_effect_of_this_catcat_from_relevances(releDict[cols],model, cols)
 for cols in model['catconts']:
  opDict[cols] = get_effect_of_this_catcont_from_relevances(releDict[cols],model,cols)
 for cols in model['contconts']:
  opDict[cols] = get_effect_of_this_contcont_from_relevances(releDict[cols],model,cols)
 return opDict
  
def pred_from_effects(base,l,contEffs,catEffs,interxEffs=None):
 op = pd.Series([base]*l)
 for col in contEffs:
  op = op*contEffs[col]
 for col in catEffs:
  op = op*catEffs[col]
 if interxEffs!=None:
  for cols in interxEffs:
   op = op*interxEffs[cols]
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

def get_effect_of_this_catcat(inputDf, model, cols):
 col1, col2 = cols.split(' X ')
 effectOfCol = pd.Series([model["catcats"][cols]["OTHER"]["OTHER"]]*len(inputDf))
 for unique1 in model['catcats'][cols]['uniques']:
  for unique2 in model['catcats'][cols]['uniques'][unique1]['uniques']:
   effectOfCol[(inputDf[col1]==unique1) & (inputDf[col2]==unique2)] = model['catcats'][cols]['uniques'][unique1]['uniques'][unique2]
  effectOfCol[(inputDf[col1]==unique1) & (~inputDf[col2].isin(model['catcats'][cols]['uniques'][unique1]['uniques']))] = model['catcats'][cols]['uniques'][unique1]['OTHER']
 for unique2 in model['catcats'][cols]['OTHER']['uniques']:
  effectOfCol[(~inputDf[col1].isin(model['catcats'][cols]['uniques'])) & (inputDf[col2]==unique2)] = model['catcats'][cols]['OTHER']['uniques'][unique2]
 return effectOfCol

def get_effect_of_this_catcont(inputDf, model, cols):
 col1, col2 = cols.split(' X ')
 x = inputDf[col2]
 effectOfCol = pd.Series([1]*len(inputDf))
 
 for unique in model['catconts'][cols]['uniques']:
  effectOfCol.loc[(inputDf[col1]==unique) & (x<=model["catconts"][cols]['uniques'][unique][0][0])] = model["catconts"][cols]['uniques'][unique][0][1] #Everything too early gets with the program
  for i in range(len(model["catconts"][cols]['uniques'][unique])-1):
   x1 = model["catconts"][cols]['uniques'][unique][i][0]
   x2 = model["catconts"][cols]['uniques'][unique][i+1][0]
   y1 = model["catconts"][cols]['uniques'][unique][i][1]
   y2 = model["catconts"][cols]['uniques'][unique][i+1][1]
   effectOfCol.loc[(inputDf[col1]==unique) & (x>=x1)&(x<=x2)] = ((x-x1)*y2 + (x2-x)*y1)/(x2 - x1)
  effectOfCol.loc[(inputDf[col1]==unique) & (x>=model["catconts"][cols]['uniques'][unique][-1][0])] = model["catconts"][cols]['uniques'][unique][-1][1] #Everything too late gets with the program
  
 effectOfCol.loc[(~inputDf[col1].isin(model['catconts'][cols]['uniques'])) & (x<=model["catconts"][cols]['OTHER'][0][0])] = model["catconts"][cols]['OTHER'][0][1] #Everything too early gets with the program
 for i in range(len(model["catconts"][cols]['OTHER'])-1):
  x1 = model["catconts"][cols]['OTHER'][i][0]
  x2 = model["catconts"][cols]['OTHER'][i+1][0]
  y1 = model["catconts"][cols]['OTHER'][i][1]
  y2 = model["catconts"][cols]['OTHER'][i+1][1]
  effectOfCol.loc[(~inputDf[col1].isin(model['catconts'][cols]['uniques'])) & (x>=x1)&(x<=x2)] = ((x-x1)*y2 + (x2-x)*y1)/(x2 - x1)
 effectOfCol.loc[(~inputDf[col1].isin(model['catconts'][cols]['uniques'])) & (x>=model["catconts"][cols]['OTHER'][-1][0])] = model["catconts"][cols]['OTHER'][-1][1] #Everything too late gets with the program
 
 return effectOfCol

def get_effect_of_this_contcont(inputDf,model,cols): #we are using x and y to predict z
 col1, col2 = cols.split(' X ')
 x = inputDf[col1]
 y = inputDf[col2]
 effectOfCol = pd.Series([1]*len(inputDf))
 
 #Corners get with the program
 
 effectOfCol.loc[(x<=model["contconts"][cols][0][0]) & (y<=model["contconts"][cols][0][1][0][0])] = model["contconts"][cols][0][1][0][1]
 effectOfCol.loc[(x>=model["contconts"][cols][-1][0]) & (y<=model["contconts"][cols][-1][1][0][0])] = model["contconts"][cols][-1][1][0][1]
 effectOfCol.loc[(x<=model["contconts"][cols][0][0]) & (y>=model["contconts"][cols][0][1][-1][0])] = model["contconts"][cols][0][1][-1][1]
 effectOfCol.loc[(x>=model["contconts"][cols][-1][0]) & (y>=model["contconts"][cols][-1][1][-1][0])] = model["contconts"][cols][-1][1][-1][1]
 
 #Edges get with the program
 
 for i in range(len(model["contconts"][cols])-1):
  x1 = model["contconts"][cols][i][0]
  x2 = model["contconts"][cols][i+1][0]
  z1 = model["contconts"][cols][i][1][0][1]
  z2 = model["contconts"][cols][i+1][1][0][1]
  effectOfCol.loc[(y<=model["contconts"][cols][i][1][0][0])&(x>=x1)&(x<=x2)] = ((x-x1)*z2 + (x2-x)*z1)/(x2 - x1)
 
 for i in range(len(model["contconts"][cols])-1):
  x1 = model["contconts"][cols][i][0]
  x2 = model["contconts"][cols][i+1][0]
  z1 = model["contconts"][cols][i][1][-1][1]
  z2 = model["contconts"][cols][i+1][1][-1][1]
  effectOfCol.loc[(y>=model["contconts"][cols][i][1][-1][0])&(x>=x1)&(x<=x2)] = ((x-x1)*z2 + (x2-x)*z1)/(x2 - x1)
 
 for i in range(len(model["contconts"][cols][0][1])-1):
  y1 = model["contconts"][cols][0][1][i][0]
  y2 = model["contconts"][cols][0][1][i+1][0]
  z1 = model["contconts"][cols][0][1][i][1]
  z2 = model["contconts"][cols][0][1][i+1][1]
  effectOfCol.loc[(x<=model["contconts"][cols][0][0])&(y>=y1)&(y<=y2)] = ((y-y1)*z2 + (y2-y)*z1)/(y2 - y1)
 
 for i in range(len(model["contconts"][cols][-1][1])-1):
  y1 = model["contconts"][cols][-1][1][i][0]
  y2 = model["contconts"][cols][-1][1][i+1][0]
  z1 = model["contconts"][cols][-1][1][i][1]
  z2 = model["contconts"][cols][-1][1][i+1][1]
  effectOfCol.loc[(x>=model["contconts"][cols][-1][0])&(y>=y1)&(y<=y2)] = ((y-y1)*z2 + (y2-y)*z1)/(y2 - y1)
 
 #The interior
 
 for i in range(len(model["contconts"][cols])-1):
  x1 = model["contconts"][cols][i][0]
  x2 = model["contconts"][cols][i+1][0]
  for j in range(len(model["contconts"][cols][i][1])-1):
   y1 = model["contconts"][cols][0][1][j][0]
   y2 = model["contconts"][cols][0][1][j+1][0]
   z11 = model["contconts"][cols][i][1][j][1]
   z12 = model["contconts"][cols][i][1][j+1][1]
   z21 = model["contconts"][cols][i+1][1][j][1]
   z22 = model["contconts"][cols][i+1][1][j+1][1]
   effectOfCol.loc[(x>=x1)&(x<=x2)&(y>=y1)&(y<=y2)] = ((x-x1)*(y-y1)*z22 + (x-x1)*(y2-y)*z21 + (x2-x)*(y-y1)*z12 + (x2-x)*(y2-y)*z11)/((x2 - x1)*(y2 - y1))
 
 return effectOfCol

def round_to_sf(x, sf=5):
 if x==0:
  return 0
 else:
  return round(x,sf-1-int(math.floor(math.log10(abs(x)))))

def roundify_cat(cat, sf=5):
 op=cat.copy()
 for k in op:
  if k=="uniques":
   for unique in op[k]:
    op[k][unique] = round(op[k][unique], sf)
  else:
   op[k]=round(op[k], sf)
 return op

def roundify_cont(cont, sf=5):
 op = copy.deepcopy(cont)
 for i in range(len(op)):
  op[i][1] = round(op[i][1],sf)
 return op

def roundify_catcat(catcat, sf=5):
 op=catcat.copy()
 for k in op:
  if k=="uniques":
   for unique in op[k]:
    op[k][unique] = roundify_cat(op[k][unique], sf)
  else:
   op[k]=roundify_cat(op[k], sf)
 return op

def roundify_catcont(catcont, sf=5):
 op=catcont.copy()
 for k in op:
  if k=="uniques":
   for unique in op[k]:
    op[k][unique] = roundify_cont(op[k][unique], sf)
  else:
   op[k]=roundify_cont(op[k], sf)
 return op

def roundify_contcont(contcont, sf=5):
 op = copy.deepcopy(contcont)
 for i in range(len(op)):
  op[i][1] = roundify_cont(op[i][1],sf)
 return op


def explain(model, sf=5):
 print("BASE_VALUE", round_to_sf(model["BASE_VALUE"], sf))
 for col in model["cats"]:
  print(col, roundify_cat(model["cats"][col], sf))
 for col in model["conts"]:
  print(col, roundify_cont(model["conts"][col], sf))
 for cols in model["catcats"]:
  print(cols, roundify_catcat(model["catcats"][col], sf))
 for cols in model["catconts"]:
  print(cols, roundify_catcont(model["catconts"][col], sf))
 for cols in model["contconts"]:
  print(cols, roundify_contcont(model["contconts"][col], sf))
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

def prep_starting_model_including_interactions(inputDf, target, cont_pts = {}, cat_uniques = {}, catcat_uniqueuniques = {}, catconts_uniquepts = {}, contconts_ptpts={}, boringValue=1, frac=1):
 
 model={"BASE_VALUE":inputDf[target].mean()*frac, "conts":{}, "cats":{}, "catcats":{}, "catconts":{}, "contconts":{}}
 
 for col in cont_pts:
  model["conts"][col]=[]
  for pt in cont_pts[col]:
   model["conts"][col].append([pt,boringValue])
 
 for col in cat_uniques:
  model["cats"][col]={"OTHER":boringValue, "uniques":{}}
  for unique in cat_uniques[col]:
   model["cats"][col]["uniques"][unique]=boringValue
 
 for cols in catcat_uniqueuniques:
  secondary = {"OTHER":boringValue, "uniques":{}}
  for unique in catcat_uniqueuniques[cols][1]:
   secondary["uniques"][unique] = boringValue
  model["catcats"][cols] = {"OTHER":secondary.copy(), "uniques":{}}
  for unique in catcat_uniqueuniques[cols][0]:
   model["catcats"][cols]['uniques'][unique] = secondary.copy()
 
 for cols in catcont_uniquepts:
  secondary = []
  for pt in catcont_uniquepts[cols][1]:
   secondary.append([pt,boringValue])
  model["catconts"][cols] = {"OTHER":secondary.copy(), "uniques":{}}
  for unique in catcont_uniquepts[cols][0]:
   model["catconts"][cols]['uniques'][unique] = secondary.copy()
 
 for cols in contcont_ptpts:
  secondary = []
  for pt in catcont_ptpts[cols][1]:
   secondary.append([pt,boringValue])
  model["catconts"][cols] = []
  for pt in contcont_ptpts[cols][0]:
   model["catconts"][cols].append([pt,secondary.copy()])
 
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
  skeys = get_sorted_keys(model['cats'][col])
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
 exampleModel = {"BASE_VALUE": 1700, 'catcats':{'cat1 X cat2':{'uniques':{'a':{'uniques':{'x':1.1,'y':1.2,'z':1.1},'OTHER':1}, 'b':{'uniques':{'x':1.1,'y':1.2,'z':1.1},'OTHER':1}, 'c':{'uniques':{'x':1.1,'y':1.2,'z':1.7},'OTHER':1}}, 'OTHER':{'uniques':{'x':1.1,'y':1.2,'z':1.1},'OTHER':1}}}, 'catconts':{'cat1 X cont1':{'uniques':{'a':[[0,0.8], [50,1.2]], 'b':[[0,0.8], [50,1.2]], 'c':[[0,0.8], [50,1.8]]}, 'OTHER':[[0,0.8], [50,1.2]]}}, 'contconts':{'cont1 X cont2': [[0,[[0,1.1], [20,1.2]]], [50,[[0,1.3], [20,1.4]]]]}}
 exampleDf = pd.DataFrame({"cont1":[0,25,50, 30], "cont2":[2,3,8,19], "cat1":["a","b","c","d"],"cat2":["w",'x','y','z'], "y":[5,7,9,11]})
 
 print(get_effect_of_this_catcat(exampleDf, exampleModel, "cat1 X cat2"))
 print(get_effect_of_this_catcont(exampleDf, exampleModel, "cat1 X cont1"))
 
 exampleDf = pd.DataFrame({"cont1":[25], "cont2":[10], 'y':[5]})
 
 print(get_effect_of_this_contcont(exampleDf, exampleModel, "cont1 X cont2"))
 
 exampleDf = pd.DataFrame({"cont1":[-999,999,-999,999], "cont2":[-999,-999,999,999], 'y':[5,5,5,5]})
 
 print(get_effect_of_this_contcont(exampleDf, exampleModel, "cont1 X cont2"))
 
 exampleDf = pd.DataFrame({"cont1":[-999,10,999,10], "cont2":[10,-999,10,999], 'y':[5,5,5,5]})
 
 print(get_effect_of_this_contcont(exampleDf, exampleModel, "cont1 X cont2"))

if False:#__name__ == '__main__':
 exampleModel = {"BASE_VALUE":1700,"conts":{"cont1":[[0.01, 1],[0.02,1.1], [0.03, 1.06]], "cont2":[[37,1.2],[98, 0.9]]}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "florpalorp":0.92}, "OTHER":1.04}}}
 exampleDf = pd.DataFrame({"cont1":[0.013,0.015,0.025, 0.035], "cont2":[37,48,45,51], "cat1":["wstfgl","florpalorp","dukis","welp"], "y":[5,7,9,11]})
 
 print(get_effect_of_this_cont_col_on_single_input(0.012, exampleModel, "cont1")) #should be 1.02
 print(get_effect_of_this_cont_col_on_single_input(0.04, exampleModel, "cont1")) #should be 1.06
 print(get_effect_of_this_cat_col_on_single_input("florpalorp", exampleModel, "cat1")) #should be 0.92
 print(get_effect_of_this_cat_col_on_single_input(12, exampleModel, "cat1")) #should be 1.04
 
 print(list(get_effect_of_this_cat_col(exampleDf, exampleModel, "cat1"))) #[1.05,0.92,1.04,1.04]
 print(list(get_effect_of_this_cont_col(exampleDf, exampleModel, "cont1"))) #[1.03,1.05,1.08,1.06]

 print(caricature_model(exampleModel,2, 0.5))
 
 

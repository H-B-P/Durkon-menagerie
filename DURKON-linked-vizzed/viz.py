import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly

def get_cont_pdp_prevalences(df, col, intervals=10, weightCol=None):
 cdf= df.copy()
 
 if type(intervals)==type([1,2,3]):
  intervs=intervals
 else:
  gap=(max(df[col])-min(df[col]))/float(intervals)
  print(min(df[col]), max(df[col]), gap)
  intervs=list(np.arange(min(df[col]), max(df[col])+gap, gap))
 
 if weightCol==None:
  cdf["weight"]=1
 else:
  cdf["weight"]=cdf[weightCol]
 
 prevs=[]
 
 for i in range(len(intervs)-1):
  loInt = intervs[i]
  hiInt = intervs[i+1]
  if i==(len(intervs)-2):
   prevs.append(sum(cdf[(cdf[col]<=hiInt) & (cdf[col]>=loInt)]["weight"]))
  else:
   prevs.append(sum(cdf[(cdf[col]<hiInt) & (cdf[col]>=loInt)]["weight"]))
 
 return intervs, prevs

def get_cat_pdp_prevalences(df, col, threshold=0.05, weightCol=None):
 cdf= df.copy()
 
 if weightCol==None:
  cdf["weight"]=1
 else:
  cdf["weight"]=cdf[weightCol]
 
 uniques = pd.unique(cdf[col])
 opDict = {}
 totalWeight = float(sum(cdf["weight"]))
 
 for unique in uniques:
  specWeight = float(sum(cdf[cdf[col]==unique]["weight"]))
  if (specWeight/totalWeight)>=threshold:
   opDict[unique] = specWeight
 
 opDict["OTHER"] = sum(cdf[~cdf[col].isin(opDict)]["weight"])
 
 return opDict



def draw_cont_pdp(pts, targetSpan=0, name="graph", model=0, boringValue=1, leeway=0.05, ytitle="Relativity"):
 X = [pt[0] for pt in pts]
 Y = [pt[1] for pt in pts]
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, boringValue-targetSpan), max(max(Y)+leeway, boringValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=go.Scatter(x=X, y=Y), layout=layout)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP Graph for "+name, xaxis_title=name, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP Graph for "+name+", model "+model, xaxis_title=name, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image("graphs/"+name+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'.html')
 else:
  fig.write_image("graphs/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'__'+model+'.html')
 




def draw_cat_pdp(dyct, targetSpan=0, name="graph", model=0, boringValue=1, leeway=0.05, ytitle="Relativity"):
 
 X=[]
 Y=[]
 for thing in dyct["uniques"]:
  X.append(thing)
  Y.append(dyct["uniques"][thing])
 #if (dyct["OTHER"]!=boringValue):
 X.append("OTHER")
 Y.append(dyct["OTHER"])
 
 print(X,Y)
 
 layout = {
  "yaxis": {
    "range": [min(min(Y)-leeway, boringValue-targetSpan), max(max(Y)+leeway, boringValue+targetSpan)]
  }
 }
 
 fig = go.Figure(data=go.Bar(x=X, y=Y), layout=layout)
 
 if name!="graph":
  if (model==0):
   fig.update_layout(title="PDP Graph for "+name, xaxis_title=name, yaxis_title=ytitle)
  else:
   fig.update_layout(title="PDP Graph for "+name+", model "+model, xaxis_title=name, yaxis_title=ytitle)
 
 if (model==0):
  fig.write_image("graphs/"+name+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'.html')
 else:
  fig.write_image("graphs/"+name+"__"+model+".png")
  plotly.offline.plot(fig, filename='graphs/'+name+'__'+model+'.html')



if __name__=="__main__":
 exampleCont = [[1,0.4],[2,0.6],[3,-0.4]]
 draw_cont_pdp(exampleCont, boringValue=0, targetSpan=0.2, ytitle="LPUs")
 #exampleCat = {"uniques":{"wstfgl":1.05, "florpalorp":0.92, "turlingdrome":0.99}, "OTHER":1.04}
 #draw_cat_pdp(exampleCat)

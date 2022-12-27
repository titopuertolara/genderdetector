import torch
import numpy as np
import json
import pandas as pd
from gravityai import gravityai as grav

def encode_2(x,emb_dict,limit):
  out=[]
  for w in x:
    rout=[emb_dict[i] for i in w]
    if len(rout)<limit:
      remain=limit-len(rout)
      dummy=[0 for _ in range(remain)]
      out.append(rout+dummy)
    else:
      out.append(rout)
  return out

model=torch.jit.load('lstm_gender_model.pt')

with open('emb_dict.json','r') as jsonfile:
  lang_dict=json.load(jsonfile)


def process(inPath,outPath):
  gender_list=['M','F']
  input_df=pd.read_csv(inPath)
  input_df['name']=input_df['name'].apply(lambda x : x.lower())
  name_list=input_df['name'].to_list()
  h_0=torch.zeros((2,len(name_list),128))
  c_0=torch.zeros((2,len(name_list),128))
  hidden=(h_0,c_0)
  input_t=torch.tensor(encode_2(name_list,lang_dict,23))
  z,_=model(input_t,hidden)
  final_pred=[gender_list[np.argmax(i.tolist())] for i in z]
  input_df['gender']=final_pred
  output_df=input_df[['name','gender']]
  output_df.to_csv(outPath,index=False)

grav.wait_for_requests(process)

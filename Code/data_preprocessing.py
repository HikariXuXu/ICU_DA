# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:34:09 2022

@author: MH Xu
"""

from tqdm import tqdm
import os
import pandas as pd
import pickle
import numpy as np

raw_data_path = '../Data'

training_label = pd.read_csv(raw_data_path+'/Outcomes-a.txt', usecols=['RecordID', 'In-hospital_death'])
training_label.RecordID = training_label.RecordID.astype(str)
training_label.rename(columns={'In-hospital_death':'in_hospital_mortality'}, inplace=True)
training_set = []
set_a_fname_list = tqdm(os.listdir(raw_data_path+'/set-a'), desc='Reading time series set a')
for f in set_a_fname_list:
    data = pd.read_csv(raw_data_path+'/set-a/'+f).iloc[1:]
    data = data.loc[data.Parameter.notna()]
    if len(data)<=5:
        continue
    data = data.loc[data.Value>=0] # neg Value indicates missingness.
    data['RecordID'] = f[:-4]
    data['label'] = int(training_label[training_label.RecordID==f[:-4]]['in_hospital_mortality'])
    training_set.append(data)
training_set = pd.concat(training_set)
training_set.Time = training_set.Time.apply(lambda x:int(x[:2])+int(x[3:])/60) # No. of hours since admission.
training_set.rename(columns={'Time':'hour', 'Parameter':'variable', 'Value':'value'}, inplace=True)




# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:45:20 2022

@author: MH Xu
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

class ReadPhysioNet():
    
    def __init__(self, dataPath, labelPath, featureScaling):
        self.dataPath = dataPath
        self.labelPath = labelPath
        self.featureScaling = featureScaling
        self.originalData = self.readData()
        self.mergedData, self.label, self.times, self.mean, self.max, self.min = self.mergeData()
        self.featureScale()
    
    
    def readData(self):
        label = pd.read_csv(self.labelPath, usecols=['RecordID', 'In-hospital_death'])
        label.RecordID = label.RecordID.astype(str)
        dataSet = []
        fnameList = tqdm(os.listdir(self.dataPath), desc='Reading time series data set', ncols=80)
        for f in fnameList:
            data = pd.read_csv(self.dataPath+'/'+f).iloc[1:]
            data = data.loc[data.Parameter.notna()]
            if len(data) <= 5:
                continue
            data = data.loc[data.Value >= 0]
            data['RecordID'] = f[:-4]
            data['label'] = int(label[label.RecordID == f[:-4]]['In-hospital_death'])
            data = self.removeOutliers(data)
            data = self.removeMechVent(data)
            dataSet.append(data)
        return dataSet
    
    
    def removeOutliers(self, data):
        # Remove outliers in pH
        data = data.drop(data[(data['Parameter']=='pH')&(data['Value']>8)].index)
        data = data.drop(data[(data['Parameter']=='pH')&(data['Value']<6)].index)
        # Remove outliers in Temp
        data = data.drop(data[(data['Parameter']=='Temp')&(data['Value']<9)].index)
        return data
    
    
    def removeMechVent(self, data):
        # Remove useless variable "MechVent"
        return data.drop(data[(data['Parameter']=='MechVent')].index)
    
    
    def mergeData(self):
        # Merge data on the same time into one row.
        dic = {'time':-1,'Age':0,'Gender':1,'Height':2,'ICUType1':3,'ICUType2':4,'ICUType3':5,'ICUType4':6,'Albumin':7,'ALP':8,'ALT':9,\
               'AST':10,'Bilirubin':11,'BUN':12,'Cholesterol':13,'Creatinine':14,'DiasABP':15,'FiO2':16,'GCS':17,'Glucose':18,'HCO3':19,\
               'HCT':20,'HR':21,'K':22,'Lactate':23,'Mg':24,'MAP':25,'Na':26,'NIDiasABP':27,'NIMAP':28,'NISysABP':29,'PaCO2':30,'PaO2':31,\
               'pH':32,'Platelets':33,'RespRate':34,'SaO2':35,'SysABP':36,'Temp':37,'TroponinI':38,'TroponinT':39,'Urine':40,'WBC':41,\
               'Weight':42}
        self.dic = dic
        sumValue = np.zeros(len(dic)-1)
        count = np.zeros(len(dic)-1)
        maxValue = np.zeros(len(dic)-1)
        maxValue[3:7] = 1
        minValue = np.zeros(len(dic)-1)
        minValue[3:7] = 0
        mergedData = []
        label = []
        times = []
        dataIndex = tqdm(np.arange(len(self.originalData)), desc='Merging time series data set', ncols=80)
        for i in dataIndex:
            obsData = []
            obsTime = []
            for index, row in self.originalData[i].iterrows():
                if row['Time'] == '00:00':
                    if 0 not in obsTime:
                        obs = [-1]*(len(dic)-1)
                        obsTime.append(0)
                        obsData.append(obs)
                    else:
                        obs = obsData[-1]
                    # One-hot encode for ICUType
                    if row['Parameter'] == 'ICUType':
                        count[dic['ICUType1']] += 1
                        count[dic['ICUType2']] += 1
                        count[dic['ICUType3']] += 1
                        count[dic['ICUType4']] += 1
                        if row['Value'] == 1:
                            sumValue[dic['ICUType1']] += 1
                            obs[dic['ICUType1']] = 1
                            obs[dic['ICUType2']] = 0
                            obs[dic['ICUType3']] = 0
                            obs[dic['ICUType4']] = 0
                        elif row['Value'] == 2:
                            sumValue[dic['ICUType2']] += 1
                            obs[dic['ICUType1']] = 0
                            obs[dic['ICUType2']] = 1
                            obs[dic['ICUType3']] = 0
                            obs[dic['ICUType4']] = 0
                        elif row['Value'] == 3:
                            sumValue[dic['ICUType3']] += 1
                            obs[dic['ICUType1']] = 0
                            obs[dic['ICUType2']] = 0
                            obs[dic['ICUType3']] = 1
                            obs[dic['ICUType4']] = 0
                        else:
                            sumValue[dic['ICUType4']] += 1
                            obs[dic['ICUType1']] = 0
                            obs[dic['ICUType2']] = 0
                            obs[dic['ICUType3']] = 0
                            obs[dic['ICUType4']] = 1
                        obsData[-1] = obs
                    else:
                        sumValue[dic[row['Parameter']]] += row['Value']
                        count[dic[row['Parameter']]] += 1
                        if row['Value'] > maxValue[dic[row['Parameter']]]:
                            maxValue[dic[row['Parameter']]] = row['Value']
                        elif row['Value'] < minValue[dic[row['Parameter']]]:
                            minValue[dic[row['Parameter']]] = row['Value']
                        obs[dic[row['Parameter']]] = row['Value']
                        obsData[-1] = obs
                else:
                    hourAndMinute=row['Time'].split(':')
                    if int(hourAndMinute[0])*60+int(hourAndMinute[1]) not in obsTime:
                        obs = [-1]*(len(dic)-1)
                        obsTime.append(int(hourAndMinute[0])*60+int(hourAndMinute[1]))
                        obs[:7] = obsData[0][:7]
                        obsData.append(obs)
                    else:
                        obs = obsData[-1]
                    sumValue[dic[row['Parameter']]] += row['Value']
                    count[dic[row['Parameter']]] += 1
                    if row['Value'] > maxValue[dic[row['Parameter']]]:
                        maxValue[dic[row['Parameter']]] = row['Value']
                    elif row['Value'] < minValue[dic[row['Parameter']]]:
                        minValue[dic[row['Parameter']]] = row['Value']
                    obs[dic[row['Parameter']]] = row['Value']
                    obsData[-1] = obs
            mergedData.append(obsData)
            label.append(row['label'])
            times.append(obsTime)
        count[count==0] = 1
        meanValue = sumValue/count
        return mergedData, label, times, meanValue, maxValue, minValue
    
    
    def featureScale(self):
        if self.featureScaling == 'Standardization':
            sumVar = np.zeros(len(self.dic)-1)
            count = np.zeros(len(self.dic)-1)
            for i in range(len(self.mergedData)):
                for j in range(7):
                    if self.mergedData[i][0][j] != -1:
                        sumVar[j] += (self.mergedData[i][0][j]-self.mean[j])**2
                        count[j] += 1
                for j in range(7,len(self.dic)-1):
                    for k in range(len(self.mergedData[i])):
                        if self.mergedData[i][k][j] != -1:
                            sumVar[j] += (self.mergedData[i][k][j]-self.mean[j])**2
                            count[j] += 1
            self.std = np.sqrt(np.array(sumVar)/np.array(count))
            dataIndex = tqdm(np.arange(len(self.mergedData)), desc='Standardizing time series data set', ncols=80)
            for i in dataIndex:
                for j in range(len(self.dic)-1):
                    for k in range(len(self.mergedData[i])):
                        if self.mergedData[i][k][j] != -1:
                            self.mergedData[i][k][j] = (self.mergedData[i][k][j]-self.mean[j])/self.std[j]
                            
        elif self.featureScaling == 'Normalization':
            dataIndex = tqdm(np.arange(len(self.mergedData)), desc='Normalizing time series data set', ncols=80)
            maxMinusMin = self.max - self.min
            for i in dataIndex:
                for j in range(len(self.dic)-1):
                    for k in range(len(self.mergedData[i])):
                        if self.mergedData[i][k][j] != -1:
                            self.mergedData[i][k][j] = (self.mergedData[i][k][j]-self.min[j])/maxMinusMin[j]
    
    
    def sliceData(self, sliceGap=60):
        # Slice merged data by slice gap.
        # Return sliceData, deltaMat, maskMat.
        self.sliceGap = sliceGap
        sliceData = []
        deltaMat = []
        maskMat = []
        dataIndex = tqdm(np.arange(len(self.mergedData)), desc='Slicing time series data set', ncols=80)
        for i in dataIndex:
            obsData = []
            delta = []
            mask = []
            lastTime = 0
            lastExistTime = np.zeros(len(self.dic)-1)
            sumExistTime = np.zeros(len(self.dic)-1)
            count = np.zeros(len(self.dic)-1)
            for j in range(len(self.mergedData[i])):
                if j == 0:
                    obsData.append(np.array(self.mergedData[i][j]))
                    mask.append((np.array(self.mergedData[i][j])!=-1)+0)
                    sumExistTime = mask[0]*self.times[i][j]
                    count = count+mask[0]
                elif j == len(self.mergedData[i])-1:
                    if self.times[i][j] < lastTime+sliceGap:
                        oneMask = (np.array(self.mergedData[i][j])!=-1)+0
                        newMask = ((mask[-1]+oneMask)>0)+0
                        obsData[-1] = obsData[-1]*mask[-1] + np.array(self.mergedData[i][j])*oneMask - np.ones(len(self.dic)-1)*(1-newMask)
                        count = count+oneMask
                        sumExistTime = sumExistTime+oneMask*self.times[i][j]
                        mask[-1] = newMask
                        count[count==0] = 1
                        obsData[-1] = obsData[-1]/count
                        lastExistTime = lastExistTime*(1-mask[-1])+sumExistTime/count
                        delta.append((lastTime+sliceGap/2-lastExistTime)*(1-mask[-1]))
                    else:
                        count[count==0] = 1
                        obsData[-1] = obsData[-1]/count
                        lastExistTime = lastExistTime*(1-mask[-1])+sumExistTime/count
                        delta.append((lastTime+sliceGap/2-lastExistTime)*(1-mask[-1]))
                        
                        if (self.times[i][j]-lastTime)//sliceGap >= 2:
                            for numGap in range((self.times[i][j]-lastTime)//sliceGap-1):
                                obsData.append(-np.ones(len(self.dic)-1))
                                obsData[-1][:7] = obsData[0][:7]
                                mask.append(np.zeros(len(self.dic)-1))
                                mask[-1][:7] = mask[0][:7]
                                delta.append((lastTime+(numGap+1.5)*sliceGap-lastExistTime)*(1-mask[-1]))
                        lastTime += ((self.times[i][j]-lastTime)//sliceGap)*sliceGap
                        obsData.append(np.array(self.mergedData[i][j]))
                        mask.append((np.array(self.mergedData[i][j])!=-1)+0)
                        sumExistTime = mask[-1]*self.times[i][j]
                        lastExistTime = lastExistTime*(1-mask[-1])+mask[-1]*self.times[i][j]
                        delta.append((lastTime+sliceGap/2-lastExistTime)*(1-mask[-1]))
                else:
                    if self.times[i][j] < lastTime+sliceGap:
                        oneMask = (np.array(self.mergedData[i][j])!=-1)+0
                        newMask = ((mask[-1]+oneMask)>0)+0
                        obsData[-1] = obsData[-1]*mask[-1] + np.array(self.mergedData[i][j])*oneMask - np.ones(len(self.dic)-1)*(1-newMask)
                        count = count+oneMask
                        sumExistTime = sumExistTime+oneMask*self.times[i][j]
                        mask[-1] = newMask
                    else:
                        count[count==0] = 1
                        obsData[-1] = obsData[-1]/count
                        lastExistTime = lastExistTime*(1-mask[-1])+sumExistTime/count
                        delta.append((lastTime+sliceGap/2-lastExistTime)*(1-mask[-1]))
                        
                        # Reset 'sumExistTime' and 'count' to zeros
                        sumExistTime = np.zeros(len(self.dic)-1)
                        count = np.zeros(len(self.dic)-1)
                        
                        if (self.times[i][j]-lastTime)//sliceGap >= 2:
                            for numGap in range((self.times[i][j]-lastTime)//sliceGap-1):
                                obsData.append(-np.ones(len(self.dic)-1))
                                obsData[-1][:7] = obsData[0][:7]
                                mask.append(np.zeros(len(self.dic)-1))
                                mask[-1][:7] = mask[0][:7]
                                delta.append((lastTime+(numGap+1.5)*sliceGap-lastExistTime)*(1-mask[-1]))
                        lastTime += ((self.times[i][j]-lastTime)//sliceGap)*sliceGap
                        obsData.append(np.array(self.mergedData[i][j]))
                        mask.append((np.array(self.mergedData[i][j])!=-1)+0)
                        sumExistTime = mask[-1]*self.times[i][j]
                        count = count+mask[-1]
            sliceData.append(obsData)
            deltaMat.append(delta)
            maskMat.append(mask)
        return sliceData, deltaMat, maskMat

if __name__ == '__main__':
    trainingData = ReadPhysioNet('E:/WashU/Research/ICU/Data/set-a', 'E:/WashU/Research/ICU/Data/Outcomes-a.txt', isNormal=1)
    sliceData, deltaMat, maskMat = trainingData.sliceData()
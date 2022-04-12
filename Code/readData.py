# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 17:45:20 2022

@author: MH Xu
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


class ReadPhysioNet():
    
    def __init__(self, trainingDataPath, trainingLabelPath, \
                 testingData1Path, testingLabel1Path, \
                 testingData2Path, testingLabel2Path, featureScaling):
        self.dic = {'time':-1,'Age':0,'Gender':1,'Height':2,'ICUType1':3,'ICUType2':4,'ICUType3':5,'ICUType4':6,'Albumin':7,'ALP':8,\
                    'ALT':9,'AST':10,'Bilirubin':11,'BUN':12,'Cholesterol':13,'Creatinine':14,'DiasABP':15,'FiO2':16,'GCS':17,\
                    'Glucose':18,'HCO3':19,'HCT':20,'HR':21,'K':22,'Lactate':23,'Mg':24,'MAP':25,'Na':26,'NIDiasABP':27,'NIMAP':28,\
                    'NISysABP':29,'PaCO2':30,'PaO2':31,'pH':32,'Platelets':33,'RespRate':34,'SaO2':35,'SysABP':36,'Temp':37,\
                    'TroponinI':38,'TroponinT':39,'Urine':40,'WBC':41,'Weight':42}
        self.featureScaling = featureScaling
        
        print('Transform Training Data...')
        self.originalTrainingData = self.readData(trainingDataPath, trainingLabelPath)
        self.mergedTrainingData, self.trainingLabel, self.trainingTimes, self.mean, self.max, self.min = self.mergeTrainingData()
        self.scaledTrainingData = self.featureScale('training')
        self.sliceTrainingData, self.trainingDeltaMat, self.trainingMaskMat = self.sliceData(self.scaledTrainingData, self.trainingTimes)
        
        print('Transform Testing Data 1...')
        self.originalTestingData1 = self.readData(testingData1Path, testingLabel1Path)
        self.mergedTestingData1, self.testingLabel1, self.testingTimes1 = self.mergeTestingData(1)
        self.scaledTestingData1 = self.featureScale('testing1')
        self.sliceTestingData1, self.testing1DeltaMat, self.testing1MaskMat = self.sliceData(self.scaledTestingData1, self.testingTimes1)
        
        print('Transform Testing Data 2...')
        self.originalTestingData2 = self.readData(testingData2Path, testingLabel2Path)
        self.mergedTestingData2, self.testingLabel2, self.testingTimes2 = self.mergeTestingData(2)
        self.scaledTestingData2 = self.featureScale('testing2')
        self.sliceTestingData2, self.testing2DeltaMat, self.testing2MaskMat = self.sliceData(self.scaledTestingData2, self.testingTimes2)
        
    
    def readData(self, dataPath, labelPath):
        label = pd.read_csv(labelPath, usecols=['RecordID', 'In-hospital_death'])
        label.RecordID = label.RecordID.astype(str)
        dataSet = []
        fnameList = tqdm(os.listdir(dataPath), desc='Reading time series data set', ncols=80)
        for f in fnameList:
            data = pd.read_csv(dataPath+'/'+f).iloc[1:]
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
    
    
    def mergeTrainingData(self):
        # Merge data on the same time into one row.
        sumValue = np.zeros(len(self.dic)-1)
        count = np.zeros(len(self.dic)-1)
        maxValue = np.zeros(len(self.dic)-1)
        maxValue[3:7] = 1
        minValue = np.zeros(len(self.dic)-1)
        minValue[3:7] = 0
        mergedData = []
        label = []
        times = []
        dataIndex = tqdm(np.arange(len(self.originalTrainingData)), desc='Merging time series training data set', ncols=80)
        for i in dataIndex:
            obsData = []
            obsTime = []
            for index, row in self.originalTrainingData[i].iterrows():
                if row['Time'] == '00:00':
                    if 0 not in obsTime:
                        obs = [-1]*(len(self.dic)-1)
                        obsTime.append(0)
                        obsData.append(obs)
                    else:
                        obs = obsData[-1]
                    # One-hot encode for ICUType
                    if row['Parameter'] == 'ICUType':
                        count[self.dic['ICUType1']] += 1
                        count[self.dic['ICUType2']] += 1
                        count[self.dic['ICUType3']] += 1
                        count[self.dic['ICUType4']] += 1
                        if row['Value'] == 1:
                            sumValue[self.dic['ICUType1']] += 1
                            obs[self.dic['ICUType1']] = 1
                            obs[self.dic['ICUType2']] = 0
                            obs[self.dic['ICUType3']] = 0
                            obs[self.dic['ICUType4']] = 0
                        elif row['Value'] == 2:
                            sumValue[self.dic['ICUType2']] += 1
                            obs[self.dic['ICUType1']] = 0
                            obs[self.dic['ICUType2']] = 1
                            obs[self.dic['ICUType3']] = 0
                            obs[self.dic['ICUType4']] = 0
                        elif row['Value'] == 3:
                            sumValue[self.dic['ICUType3']] += 1
                            obs[self.dic['ICUType1']] = 0
                            obs[self.dic['ICUType2']] = 0
                            obs[self.dic['ICUType3']] = 1
                            obs[self.dic['ICUType4']] = 0
                        else:
                            sumValue[self.dic['ICUType4']] += 1
                            obs[self.dic['ICUType1']] = 0
                            obs[self.dic['ICUType2']] = 0
                            obs[self.dic['ICUType3']] = 0
                            obs[self.dic['ICUType4']] = 1
                        obsData[-1] = obs
                    else:
                        sumValue[self.dic[row['Parameter']]] += row['Value']
                        count[self.dic[row['Parameter']]] += 1
                        if row['Value'] > maxValue[self.dic[row['Parameter']]]:
                            maxValue[self.dic[row['Parameter']]] = row['Value']
                        elif row['Value'] < minValue[self.dic[row['Parameter']]]:
                            minValue[self.dic[row['Parameter']]] = row['Value']
                        obs[self.dic[row['Parameter']]] = row['Value']
                        obsData[-1] = obs
                else:
                    hourAndMinute=row['Time'].split(':')
                    if int(hourAndMinute[0])*60+int(hourAndMinute[1]) not in obsTime:
                        obs = [-1]*(len(self.dic)-1)
                        obsTime.append(int(hourAndMinute[0])*60+int(hourAndMinute[1]))
                        obs[:7] = obsData[0][:7]
                        obsData.append(obs)
                    else:
                        obs = obsData[-1]
                    sumValue[self.dic[row['Parameter']]] += row['Value']
                    count[self.dic[row['Parameter']]] += 1
                    if row['Value'] > maxValue[self.dic[row['Parameter']]]:
                        maxValue[self.dic[row['Parameter']]] = row['Value']
                    elif row['Value'] < minValue[self.dic[row['Parameter']]]:
                        minValue[self.dic[row['Parameter']]] = row['Value']
                    obs[self.dic[row['Parameter']]] = row['Value']
                    obsData[-1] = obs
            mergedData.append(obsData)
            label.append(row['label'])
            times.append(obsTime)
        count[count==0] = 1
        meanValue = sumValue/count
        return mergedData, label, times, meanValue, maxValue, minValue
    
    
    def mergeTestingData(self, testNum):
        if testNum == 1:
            originalData = self.originalTestingData1
        else:
            originalData = self.originalTestingData2
        # Merge data on the same time into one row.
        mergedData = []
        label = []
        times = []
        dataIndex = tqdm(np.arange(len(originalData)), desc='Merging time series testing data set', ncols=80)
        for i in dataIndex:
            obsData = []
            obsTime = []
            for index, row in originalData[i].iterrows():
                if row['Time'] == '00:00':
                    if 0 not in obsTime:
                        obs = [-1]*(len(self.dic)-1)
                        obsTime.append(0)
                        obsData.append(obs)
                    else:
                        obs = obsData[-1]
                    # One-hot encode for ICUType
                    if row['Parameter'] == 'ICUType':
                        if row['Value'] == 1:
                            obs[self.dic['ICUType1']] = 1
                            obs[self.dic['ICUType2']] = 0
                            obs[self.dic['ICUType3']] = 0
                            obs[self.dic['ICUType4']] = 0
                        elif row['Value'] == 2:
                            obs[self.dic['ICUType1']] = 0
                            obs[self.dic['ICUType2']] = 1
                            obs[self.dic['ICUType3']] = 0
                            obs[self.dic['ICUType4']] = 0
                        elif row['Value'] == 3:
                            obs[self.dic['ICUType1']] = 0
                            obs[self.dic['ICUType2']] = 0
                            obs[self.dic['ICUType3']] = 1
                            obs[self.dic['ICUType4']] = 0
                        else:
                            obs[self.dic['ICUType1']] = 0
                            obs[self.dic['ICUType2']] = 0
                            obs[self.dic['ICUType3']] = 0
                            obs[self.dic['ICUType4']] = 1
                        obsData[-1] = obs
                    else:
                        obs[self.dic[row['Parameter']]] = row['Value']
                        obsData[-1] = obs
                else:
                    hourAndMinute=row['Time'].split(':')
                    if int(hourAndMinute[0])*60+int(hourAndMinute[1]) not in obsTime:
                        obs = [-1]*(len(self.dic)-1)
                        obsTime.append(int(hourAndMinute[0])*60+int(hourAndMinute[1]))
                        obs[:7] = obsData[0][:7]
                        obsData.append(obs)
                    else:
                        obs = obsData[-1]
                    obs[self.dic[row['Parameter']]] = row['Value']
                    obsData[-1] = obs
            mergedData.append(obsData)
            label.append(row['label'])
            times.append(obsTime)
        return mergedData, label, times
    
    
    def featureScale(self, datasetType):
        if self.featureScaling == 'Standardization':
            if datasetType == 'training':
                mergedData = self.mergedTrainingData
                sumVar = np.zeros(len(self.dic)-1)
                count = np.zeros(len(self.dic)-1)
                for i in range(len(mergedData)):
                    for j in range(7):
                        if mergedData[i][0][j] != -1:
                            sumVar[j] += (mergedData[i][0][j]-self.mean[j])**2
                            count[j] += 1
                    for j in range(7,len(self.dic)-1):
                        for k in range(len(mergedData[i])):
                            if mergedData[i][k][j] != -1:
                                sumVar[j] += (mergedData[i][k][j]-self.mean[j])**2
                                count[j] += 1
                self.std = np.sqrt(np.array(sumVar)/np.array(count))
            elif datasetType == 'testing1':
                mergedData = self.mergedTestingData1
            elif datasetType == 'testing2':
                mergedData = self.mergedTestingData2
            dataIndex = tqdm(np.arange(len(mergedData)), desc='Standardizing time series data set', ncols=80)
            for i in dataIndex:
                for j in range(len(self.dic)-1):
                    for k in range(len(mergedData[i])):
                        if mergedData[i][k][j] != -1:
                            mergedData[i][k][j] = (mergedData[i][k][j]-self.mean[j])/self.std[j]
            return mergedData
                            
        elif self.featureScaling == 'Normalization':
            if datasetType == 'training':
                mergedData = self.mergedTrainingData
            elif datasetType == 'testing1':
                mergedData = self.mergedTestingData1
            elif datasetType == 'testing2':
                mergedData = self.mergedTestingData2
            dataIndex = tqdm(np.arange(len(mergedData)), desc='Normalizing time series data set', ncols=80)
            maxMinusMin = self.max - self.min
            for i in dataIndex:
                for j in range(len(self.dic)-1):
                    for k in range(len(mergedData[i])):
                        if mergedData[i][k][j] != -1:
                            mergedData[i][k][j] = (mergedData[i][k][j]-self.min[j])/maxMinusMin[j]
            return mergedData
        
        else:
            if datasetType == 'training':
                return self.mergedTrainingData
            elif datasetType == 'testing1':
                return self.mergedTestingData1
            elif datasetType == 'testing2':
                return self.mergedTestingData2
    
    
    def sliceData(self, inputData, times, sliceGap=60):
        # Slice merged data by slice gap.
        # Return sliceData, deltaMat, maskMat.
        self.sliceGap = sliceGap
        sliceData = []
        deltaMat = []
        maskMat = []
        dataIndex = tqdm(np.arange(len(inputData)), desc='Slicing time series data set', ncols=80)
        for i in dataIndex:
            obsData = []
            delta = []
            mask = []
            lastTime = 0
            lastExistTime = np.zeros(len(self.dic)-1)
            sumExistTime = np.zeros(len(self.dic)-1)
            count = np.zeros(len(self.dic)-1)
            for j in range(len(inputData[i])):
                if j == 0:
                    obsData.append(np.array(inputData[i][j]))
                    mask.append((np.array(inputData[i][j])!=-1)+0)
                    sumExistTime = mask[0]*times[i][j]
                    count = count+mask[0]
                elif j == len(inputData[i])-1:
                    if times[i][j] <= lastTime+sliceGap:
                        oneMask = (np.array(inputData[i][j])!=-1)+0
                        newMask = ((mask[-1]+oneMask)>0)+0
                        obsData[-1] = obsData[-1]*mask[-1] + np.array(inputData[i][j])*oneMask - np.ones(len(self.dic)-1)*(1-newMask)
                        count = count+oneMask
                        sumExistTime = sumExistTime+oneMask*times[i][j]
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
                        
                        if (times[i][j]-lastTime-1)//sliceGap >= 2:
                            for numGap in range((times[i][j]-lastTime)//sliceGap-1):
                                obsData.append(-np.ones(len(self.dic)-1))
                                obsData[-1][:7] = obsData[0][:7]
                                mask.append(np.zeros(len(self.dic)-1))
                                mask[-1][:7] = mask[0][:7]
                                delta.append((lastTime+(numGap+1.5)*sliceGap-lastExistTime)*(1-mask[-1]))
                        lastTime += ((times[i][j]-lastTime-1)//sliceGap)*sliceGap
                        obsData.append(np.array(inputData[i][j]))
                        mask.append((np.array(inputData[i][j])!=-1)+0)
                        sumExistTime = mask[-1]*times[i][j]
                        lastExistTime = lastExistTime*(1-mask[-1])+mask[-1]*times[i][j]
                        delta.append((lastTime+sliceGap/2-lastExistTime)*(1-mask[-1]))
                        
                        if lastTime != 48*60-sliceGap:
                            for numGap in range((48*60-lastTime-1)//sliceGap):
                                obsData.append(-np.ones(len(self.dic)-1))
                                obsData[-1][:7] = obsData[0][:7]
                                mask.append(np.zeros(len(self.dic)-1))
                                mask[-1][:7] = mask[0][:7]
                                delta.append((lastTime+(numGap+1.5)*sliceGap-lastExistTime)*(1-mask[-1]))
                                
                else:
                    if times[i][j] <= lastTime+sliceGap:
                        oneMask = (np.array(inputData[i][j])!=-1)+0
                        newMask = ((mask[-1]+oneMask)>0)+0
                        obsData[-1] = obsData[-1]*mask[-1] + np.array(inputData[i][j])*oneMask - np.ones(len(self.dic)-1)*(1-newMask)
                        count = count+oneMask
                        sumExistTime = sumExistTime+oneMask*times[i][j]
                        mask[-1] = newMask
                    else:
                        count[count==0] = 1
                        obsData[-1] = obsData[-1]/count
                        lastExistTime = lastExistTime*(1-mask[-1])+sumExistTime/count
                        delta.append((lastTime+sliceGap/2-lastExistTime)*(1-mask[-1]))
                        
                        # Reset 'sumExistTime' and 'count' to zeros
                        sumExistTime = np.zeros(len(self.dic)-1)
                        count = np.zeros(len(self.dic)-1)
                        
                        if (times[i][j]-lastTime-1)//sliceGap >= 2:
                            for numGap in range((times[i][j]-lastTime-1)//sliceGap-1):
                                obsData.append(-np.ones(len(self.dic)-1))
                                obsData[-1][:7] = obsData[0][:7]
                                mask.append(np.zeros(len(self.dic)-1))
                                mask[-1][:7] = mask[0][:7]
                                delta.append((lastTime+(numGap+1.5)*sliceGap-lastExistTime)*(1-mask[-1]))
                        lastTime += ((times[i][j]-lastTime-1)//sliceGap)*sliceGap
                        obsData.append(np.array(inputData[i][j]))
                        mask.append((np.array(inputData[i][j])!=-1)+0)
                        sumExistTime = mask[-1]*times[i][j]
                        count = count+mask[-1]
            sliceData.append(obsData)
            deltaMat.append(delta)
            maskMat.append(mask)
        return sliceData, deltaMat, maskMat



data = ReadPhysioNet('E:/WashU/Research/ICU/Data/set-a', 'E:/WashU/Research/ICU/Data/Outcomes-a.txt', 
                     'E:/WashU/Research/ICU/Data/set-b', 'E:/WashU/Research/ICU/Data/Outcomes-b.txt', 
                     'E:/WashU/Research/ICU/Data/set-c', 'E:/WashU/Research/ICU/Data/Outcomes-c.txt', featureScaling='Normalization')

with open('E:/WashU/Research/ICU/Data/ReadPhysioNet.txt', 'wb') as f:
    pickle.dump(data, f)
    f.close()
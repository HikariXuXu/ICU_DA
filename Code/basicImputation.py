# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def imputeMean(scliedData, MaskMat, meanvalue):
    imputedata = []
    for i in range(len(scliedData)):
        a = np.array(scliedData[i])
        b = np.array(scliedData[i]) * np.array(MaskMat[i])
        e = b.sum(axis=0)
        c = np.array(MaskMat[0])
        d = c.sum(axis=0)
        d[d==0]=1
        for j in range(a.shape[1]):
            if np.all(a[:,j] == -1):
                a[:,j]= meanvalue[j]
            else:
                k = a[:,j]
                k[k==-1]= e[j]
        imputedata.append(a)
    return imputedata

def imputeNearest(sliceData, meanvalue):
    imputedata = []
    for i in range(len(sliceData)):
        a = np.array(sliceData[i])
        b = pd.DataFrame(np.array(a))
        b = b.replace(-1,np.nan)
        c = b.count()
        for k in range(a.shape[1]):
            if c[k] == 0:
                b[k] = meanvalue[k]
            if c[k] == 1:
                b[k] = b[k].interpolate(method ='ffill',axis=0)
                b[k] = b[k].interpolate(method ='bfill',axis=0)
        b = b.interpolate(method ='nearest',axis=0,limit_direction ='both')
        b = b.interpolate(method ='ffill')
        b = b.interpolate(method ='bfill')
        b = b.values
        imputedata.append(b)
    return imputedata

def imputeLast(sliceData, meanvalue):
    imputedata = []
    for i in range(len(sliceData)):
        a = np.array(sliceData[i])
        for j in range(a.shape[1]):
            if np.all(a[:,j] == -1):
                a[:,j]= meanvalue[j]
        b = pd.DataFrame(np.array(a))
        b = b.replace(-1,np.nan)
        b = b.interpolate(method ='ffill',axis=0)
        b = b.interpolate(method ='bfill',axis=0)
        b = b.values
        imputedata.append(b)
    return imputedata

def rearrange_data(imputedata):
    rearrange_data = []
    for i in range(len(imputedata)):
        a = imputedata[i]
        b = a.reshape(1,-1)
        index=[]
        for j in range (47):
            indexes = [43*(j+1), 43*(j+1)+1, 43*(j+1)+2, 43*(j+1)+3, 43*(j+1)+4, 43*(j+1)+5, 43*(j+1)+6]
            index = np.append(index,indexes)
            index = index.reshape(1,-1)
            index = index.astype(int)
        b = np.delete(b, index)
        rearrange_data = np.append(rearrange_data,b)
    rearrange_data = rearrange_data.reshape(len(imputedata),-1)
    return rearrange_data
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys

if len(sys.argv)!=2:
   sys.exit(sys.argv[0]+"[num particles]")

print("maketraininference: Program is agnostic on which units the user chooses, but energy are expected to be normalizied for the total number of particles yet.")

os.mkdir("dataset")
os.mkdir("dataset/inference")

N=int(sys.argv[1])

el='descriptor_dataset.csv'
dataset=pd.read_csv(el,dtype='float64')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,ndes))
data_ts=dataset
nf_ts=np.shape(data_ts)[0]

np.save("dataset/inference/descriptors.npy",data_ts)
del dataset 
del data_ts

el='radialderivat_dataset.csv'
dataset=pd.read_csv(el,dtype='float64')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,3,ndes))
data_ts=dataset

np.save("dataset/inference/intder2b.npy",data_ts)
del dataset
del data_ts


el='radialint_dataset.csv'
dataset=pd.read_csv(el,dtype='int32')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,ndes))
data_ts=dataset
np.save("dataset/inference/int2b.npy",data_ts)
del dataset
del data_ts


el='tripletsint_dataset.csv'
dataset=pd.read_csv(el,dtype='int32')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,ndes))
data_ts=dataset

np.save("dataset/inference/int3b.npy",data_ts)
del dataset
del data_ts

el='tripletsderivat_dataset.csv'
dataset=pd.read_csv(el,dtype='float64')
dataset = dataset.iloc[:,:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1,N,3,ndes))
data_ts=dataset
np.save("dataset/inference/intder3b.npy",data_ts)
del dataset
del data_ts

try:
    el='dataset_energy.dat'
    dataset=pd.read_csv(el,dtype='float64')
    dataset = dataset.iloc[:].values
    ndes=dataset.shape[1]
    dataset=np.reshape(dataset,(-1))
    data_ts=dataset
    np.save("dataset/inference/energy.npy",data_ts)
    del dataset
    del data_ts
except:
    print("No dataset on energies has been found. I expect you want to run a force-matching procedure. If not check the energy path")
    np.save("dataset/inference/energy.npy",np.zeros((nf_ts)))

el='dataset_force.dat'
dataset=pd.read_csv(el,dtype='float64',sep=' ')
dataset = dataset.iloc[:,:].values
dataset=np.reshape(dataset,(-1,N,3))
data_ts=dataset
np.save("dataset/inference/force.npy",data_ts)
del dataset
del data_ts


el='dataset_position.dat'
dataset=np.loadtxt(el,dtype='float64')
dataset=dataset.reshape((-1,5+3*N))
data_ts=dataset
np.save("dataset/inference/position.npy",data_ts)
del dataset
del data_ts


el='dataset_pressure.dat'
dataset=pd.read_csv(el,dtype='float64')
dataset = dataset.iloc[:].values
ndes=dataset.shape[1]
dataset=np.reshape(dataset,(-1))
data_ts=dataset
np.save("dataset/inference/pressure.npy",data_ts)
del dataset
del data_ts


#!/home/francegm/miniconda3/envs/fsenv/bin/python3
#IMPORT NN and math libraries
import tensorflow as tf
import argparse
import sys
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input
import numpy as np
import os
import shutil as sh
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
####ISTRUZIONI DA RIGA DI COMANDO
parser = argparse.ArgumentParser()
#########DATASET DIMENSION PARAMETERS##########
parser.add_argument("-N", help="number of particles")
parser.add_argument("-num_des", help="number of descriptors for particle")
parser.add_argument("-imodel", help="model to save")
###OUTPUT PARAMETER
parser.add_argument("-modelname", help="indicate a path/name for the exported model (e.g. folder/folder2/namemodel")


##Read shape parameter for keras model
args = parser.parse_args()
N=int(args.N)
num_des=int(args.num_des)
namemodel=args.modelname
input_model=args.imodel
mean=np.loadtxt(input_model+'/alpha_mu.dat')
##Build class for graph process execution
class TestModel(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(name='input1',shape=(1,N,num_des), dtype=tf.float64)])
    def testmodel(self,des):
        self.model=newmodel
        logdes=tf.math.log(des+10**(-3))-mean
        Energy=self.model(logdes)
        gradEn=tf.reshape(tf.gradients(Energy,des),shape=(N,num_des))
        return  Energy,gradEn;





with tf.device('/cpu:0'):
     model=tf.keras.models.load_model(input_model+'/net_model_type0')
newmodel=tf.keras.Sequential()
newmodel.add(Input(shape=(N,num_des,)))
for el in model.layers:
    newmodel.add(el)






##Call Class to build the graph
toexport=TestModel()

###Save the model
tf.saved_model.save(toexport, namemodel+'/model')
sh.copy(input_model+'/alpha_2body.dat',namemodel)
sh.copy(input_model+'/alpha_3body.dat',namemodel)
sh.copy(input_model+'/model_error',namemodel)

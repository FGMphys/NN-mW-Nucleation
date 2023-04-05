#!/usr/bin/env python3

#IMPORT I/O file
import os
import time
import sys
import tensorflow as tf

from numpy.random import seed
import matplotlib.pyplot as plt


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from numpy.random import default_rng



import numpy as np
import yaml

from gradient_utility import gradforceop_triplNODEN
from gradient_utility import gradforceop_rNODEN
from gradient_utility import grad_twobody_par
from gradient_utility import grad_threebody_par

from full_model_alphanes.alpha_nes_full_mod import alpha_nes_full
from full_model_alphanes.alpha_nes_full_mod import alpha_nes_full_notype

from source_routine.physics_layer_mod import physics_layer
from source_routine.physics_layer_mod import lognorm_layer

from optimizer_learning_rate_utility import build_learning_rate
from optimizer_learning_rate_utility import build_optimizer


from source_routine.force_layer_mod import force_layer

@tf.function()
def MSE(ypred,y):
   loss_function=tf.reduce_mean(tf.square((ypred-y)))
   return loss_function



def make_dataset_stream(base_pattern,mode):
    energy_on_disk=np.load(base_pattern+'/'+mode+'/'+'energy.npy',mmap_mode='r')
    force_on_disk=np.load(base_pattern+'/'+mode+'/'+'force.npy',mmap_mode='r')

    descri_on_disk=np.load(base_pattern+'/'+mode+'/'+'descriptors.npy',mmap_mode='r')
    des3bsupp_on_disk=np.load(base_pattern+'/'+mode+'/'+'des3bsupp.npy',mmap_mode='r')

    int2b_on_disk=np.load(base_pattern+'/'+mode+'/'+'int2b.npy',mmap_mode='r')
    int3b_on_disk=np.load(base_pattern+'/'+mode+'/'+'int3b.npy',mmap_mode='r')

    intder2b_on_disk=np.load(base_pattern+'/'+mode+'/'+'intder2b.npy',mmap_mode='r')
    intder3b_on_disk=np.load(base_pattern+'/'+mode+'/'+'intder3b.npy',mmap_mode='r')
    intder3bsupp_on_disk=np.load(base_pattern+'/'+mode+'/'+'intder3bsupp.npy',mmap_mode='r')


    return energy_on_disk,force_on_disk,descri_on_disk,des3bsupp_on_disk,int2b_on_disk,int3b_on_disk,intder2b_on_disk,intder3b_on_disk,intder3bsupp_on_disk

def check_dimension(buffdim,dimension,mode):
    res=buffdim
    if buffdim>dimension:
       print("alpha_nes: buffdim in ",mode," mode is bigger than number of frames in the dataset. We set buddim=datasetdim!")
       res=dimension
    return res
def make_idx_str(dimension,buffdim,mode):
    buffdim=check_dimension(buffdim,dimension,mode)
    truedim=dimension//buffdim*buffdim
    rejected=dimension%buffdim
    print("\nalpha_nes: It will be rejected ",rejected,' frames picked randomly to ensure batch size and buffer requested.\n')
    vec=np.arange(0,dimension)
    np.random.shuffle(vec)
    vec=np.reshape(vec[:truedim],(dimension//buffdim,buffdim))
    return buffdim,vec

def check_along_frames(list_of_arr,axis):
    ref=list_of_arr[0].shape[axis]
    for el in list_of_arr:
        if ref!=el.shape[axis]:
           sys.exit("Dataset are not valid. Error on dimension along axis "+str(axis))
    return 0




def force_criterio(val_tot,val_f,bestval):
    newbest=0
    if val_f<bestval:
       res=1
       newbest=val_f
    else:
       res=0
    return res,newbest
def force_energy_criterio(val_tot,val_f,bestval):
    newbest=0
    if val_tot<bestval:
       res=1
       newbest=val_tot
    else:
       res=0
    return res,newbest
def is_improved(val_p,best):
    newbest=0
    if val_p<bestvirial:
       res=1
       newbest=val_p
    else:
       res=0
    return res,newbest
def make_typemap(tipos):
    num=0
    list_tmap=[]
    for el in tipos:
        for k in range(el):
            list_tmap.append(num)
        num=num+1
    return list_tmap
################# MAIN #########################################################
##Read the input file
with open(sys.argv[1]) as file:
    full_param = yaml.load(file, Loader=yaml.FullLoader)

#Set seed
try:
    seed_par=int(full_param['Seed'])
    seed(seed_par)
    tf.random.set_seed(seed_par+1)
    print("alpha_nes: seed fixed to custom value ", seed_par,end='\n')
except:
    seed_par=12345
    seed(seed_par)
    tf.random.set_seed(seed_par+1)
    print("alpha_nes: seed fixed by default 12345\n")
#Read dataset map on disk
base_pattern=full_param['dataset_folder']
try:
    tipos=np.loadtxt(base_pattern+"/type.dat",dtype=int).reshape(-1,1)
    if tipos.shape[0]>1:
       tipos=[n_per_type for n_per_type in tipos[:,0]]
       type_map=make_typemap(tipos)
       np.savetxt('type_map.dat',np.array(type_map,dtype='int'))
    else:
       tipos=[tipos[0,0]]
       type_map=make_typemap(tipos)

except:
    sys.exit("alpha_nes: In the dataset folder it is expected to have a type.dat file with the code for the atom type!")
[e_map_tr,f_map_tr,des_map_tr,des3bsupp_map_tr,int2b_map_tr,int3b_map_tr,
intder2b_map_tr,intder3b_map_tr,intder3bsupp_map_tr]=make_dataset_stream(base_pattern,'training')
[e_map_ts,f_map_ts,des_map_ts,des3bsupp_map_ts,int2b_map_ts,int3b_map_ts,intder2b_map_ts,
intder3b_map_ts,intder3bsupp_map_ts]=make_dataset_stream(base_pattern,'test')
###Check dimension of dataset
check_along_frames([e_map_tr,f_map_tr,des_map_tr,des3bsupp_map_tr,int2b_map_tr,
int3b_map_tr,intder2b_map_tr,intder3b_map_tr,intder3bsupp_map_tr],0)
check_along_frames([e_map_ts,f_map_ts,des_map_ts,des3bsupp_map_ts,int2b_map_ts,
int3b_map_ts,intder2b_map_ts,intder3b_map_ts,intder3bsupp_map_ts],0)
#Building a stream vector
buffer_stream_tr=full_param['buffer_stream_dim_tr']
buffer_stream_ts=full_param['buffer_stream_dim_ts']

subsamp=full_param['subsampling']
if subsamp!='no':
   dimtr=int(subsamp.split()[0])
   dimts=int(subsamp.split()[1])
else:
   dimtr=des_map_tr.shape[0]
   dimts=des_map_ts.shape[0]
[buffer_stream_tr,idx_str_tr]=make_idx_str(dimtr,buffer_stream_tr,'train')
[buffer_stream_ts,idx_str_ts]=make_idx_str(dimts,buffer_stream_ts,'test')

###Initialize alphas
nalpha_r=int(full_param['dimension_encoder_2body'])
nalpha_a=int(full_param['dimension_encoder_3body'])




### Loop parameters
ne=int(full_param['number_of_epochs'])
bs=int(full_param['batch_size'])
nb=idx_str_tr.shape[1]//bs+idx_str_tr.shape[1]%bs


### Building Net parameters
actfun=full_param['activation_function']
nhl=full_param['number_of_decoding_layers']
if nhl>0:
   nD=[int(k) for k in full_param['number_of_decoding_nodes'].split()]
else:
   nD=0
###Initialize the Encoder and Decoder
tf.keras.backend.set_floatx('float64')

##Building the learning rate and then the optimizer
lr_net_param=full_param['lr_dense_net'].split()
lr_net=build_learning_rate(lr_net_param,ne,nb,idx_str_tr.shape[0],'net')

opt_net_param=full_param['optimizer_net'].split()
opt_net=build_optimizer(opt_net_param,lr_net)

lr_phys_param=full_param['lr_phys_net'].split()
lr_phys=build_learning_rate(lr_phys_param,ne,nb,idx_str_tr.shape[0],'phys')

opt_phys_param=full_param['optimizer_phys'].split()
opt_phys=build_optimizer(opt_phys_param,lr_phys)


#Composing the model

try:
    alpha_bound=float(full_param['alpha_bound'])
    print("alpha_nes: alphas will be upper-bound to custom",alpha_bound,sep=' ',end='\n')
except:
    alpha_bound=1.
    print("alpha_nes: alphas will be upper-bound to default",alpha_bound,sep=' ',end='\n')


bound=alpha_bound
limit=bound
limit3b=bound
init_alpha2b=np.random.rand(nalpha_r)*2*limit-limit
vec=np.zeros((nalpha_a,3))
vec[:,:2]=(np.random.rand(nalpha_a*2)*2*limit3b-limit3b).reshape(nalpha_a,2)
vec[:,2]=(np.random.rand(nalpha_a)*-10).reshape(nalpha_a)
init_alpha3b=vec
nt=len(tipos)
nt_couple=int(nt+nt*(nt-1)/2)
if nt!=1:
   initial_type_emb_2b=(np.random.rand(nt)*5.).reshape(nt)
   initial_type_emb_3b=(np.random.rand(nt_couple)*5.).reshape(nt_couple)
else:
   initial_type_emb_2b=np.ones(nt)
   initial_type_emb_3b=np.ones(nt_couple)
initial_type_emb=[initial_type_emb_2b,initial_type_emb_3b]


Physics_Layer=physics_layer(init_alpha2b,init_alpha3b,
                            initial_type_emb)


init_mu=np.random.rand(nalpha_r+nalpha_a)*2*limit-limit
Lognorm_Layer=lognorm_layer(nalpha_r+nalpha_a,init_mu)

nr=int2b_map_tr.shape[-1]-1

Force_Layer=force_layer()



try:
    loss_meth=full_param['loss_method']
    if loss_meth=='huber':
       HUBER = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
       model_loss=HUBER
       val_loss=MSE
       print("alpha_nes: the loss function is HUBER and validation loss is MSE.")
    else:
       model_loss=MSE
       val_loss=MSE
       print("alpha_nes: the loss function is MSE  as the validation loss!")
except:
    model_loss=MSE
    val_loss=MSE
    print("alpha_nes: the loss function is MSE loss as the validation loss!")

##METTERE PRESSURE LAYER NEL MODELLO E VALUTARLO SOLO NELLA VALIDATION (ogni epoca)

restart_par=full_param['restart']
if nt==1:
   model=alpha_nes_full_notype(Physics_Layer,Force_Layer,nhl,nD,actfun,1,model_loss,
                     val_loss,opt_net,opt_phys,alpha_bound,Lognorm_Layer,tipos,
                     type_map)
   model.restart(restart_par)
   print("alpha_nes: one atom type system has been detected. Model will be built with no atom type embedding")
else:
     model=alpha_nes_full(Physics_Layer,Force_Layer,nhl,nD,actfun,1,model_loss,
                     val_loss,opt_net,opt_phys,alpha_bound,Lognorm_Layer,tipos,
                     type_map)
     model.restart(restart_par)
try:
   train_meth=full_param['type_of_training']
except:
   train_meth='energy+force'
if train_meth=='energy+force':
   trainmeth=model.full_train_e_f
   testmeth=model.full_test_e_f
   print("alpha_nes: training will be on both energies and forces")
elif train_meth=='energy':
     trainmeth=model.full_train_e
     testmeth=model.full_test_e
     print("alpha_nes: training will be on  energies only")
else:
    sys.exit("alpha_nes: Error in type_of_training key. Possible choices are energy+force or energy")

bestval=10**5
if restart_par!='no':
   fileOU=open('lcurve.out','a')
   print("alpha_nes: learning curve will be appended to the previous one.")
else:
   fileOU=open('lcurve.out','w')
   print("#Val_loss_e    #Val_loss_f  #loss_tot    #lr_phys #lr_net\n",file=fileOU)
   out_time=open("time_story.dat",'w')
   print("#Time per epoch training  #Time per epoch test\n",file=out_time)
model_name=full_param['model_name']
if restart_par=='no':
    try:
        os.mkdir(model_name)
    except:
        os.mkdir(model_name+'new')
        model_name=model_name+'new'

gl_step=0

#lossfunction prefactors
try:
    pe=tf.constant(float(full_param['loss_energy_prefactor']),dtype='float64')
    pf=tf.constant(float(full_param['loss_force_prefactor']),dtype='float64')
    print("alpha_nes: pe and pf set to custom values",pe.numpy(),pf.numpy(),sep=' ',end='\n')
except:
    pe=tf.constant(1.,dtype='float64')
    pf=tf.constant(1.,dtype='float64')
    print("alpha_nes: pe and pf set to default value 1 1",sep=' ',end='\n')
if restart_par=='no':
   Physics_Layer.savealphas(model_name,"initial_")
   Lognorm_Layer.savemu(model_name,"initial_")
try:
   save_mod=str(full_param['save_mode'])
except:
   save_mod='other'

if save_mod=='forces':
   criterio=force_criterio
   print("alpha_nes: best model will be selected on only force loss",end='\n')
else:
   criterio=force_energy_criterio
   print("alpha_nes: best model will be selected on both energy and force error",end='\n')




lr_file=open("lr_step.dat",'a')
for ep in range(ne):
    start=time.time()
    losstot=tf.constant(0.,dtype='float64')
    vallosstot=tf.constant(0.,dtype='float64')
    vallosstote=tf.constant(0.,dtype='float64')
    vallosstotf=tf.constant(0.,dtype='float64')
    for numbuf,el in enumerate(idx_str_tr):
        loss_buffer=0.
        datasetcomplete = tf.data.Dataset.from_tensor_slices((des_map_tr[el],des3bsupp_map_tr[el],e_map_tr[el],f_map_tr[el],
                                                              intder2b_map_tr[el],int2b_map_tr[el],
                                                              intder3b_map_tr[el],intder3bsupp_map_tr[el],
                                                              int3b_map_tr[el],)).batch(bs,drop_remainder=False)
        for step, (x,x3bsupp,etrue,ftrue,intder2b,int2b,intder3b,intder3bsupp,int3b) in enumerate(datasetcomplete):
            [loss,lossf,losse,prova,provaa]=trainmeth(x,x3bsupp,int2b,intder2b,int3b,intder3b,
                                         intder3bsupp,etrue,ftrue,pe,pf)

            gl_step=gl_step+1
            lrnow=lr_phys(gl_step)
            lrnow2=lr_net(gl_step)
            lr_file.write(str(lrnow.numpy())+'\n')
            lr_file.flush()
            loss_buffer+=loss
        losstot+=loss_buffer
    losstot*=1/(step+1)/(numbuf+1)
    stop_tr=time.time()
    for stepts,el in enumerate(idx_str_ts):
        datasetcomplete = tf.data.Dataset.from_tensor_slices((des_map_ts[el],des3bsupp_map_ts[el],e_map_ts[el],f_map_ts[el],
                                                              intder2b_map_ts[el],int2b_map_ts[el],
                                                              intder3b_map_ts[el],intder3bsupp_map_ts[el],
                                                              int3b_map_ts[el])).batch(buffer_stream_ts)
        for step2, (x,x3bsupp,etrue,ftrue,intder2b,int2b,intder3b,intder3bsupp,int3b) in enumerate(datasetcomplete):
            [val_loss,val_lossf,val_losse]=testmeth(x,x3bsupp,int2b,intder2b,
                                                    int3b,intder3b,intder3bsupp,
                                                    etrue,ftrue)
            vallosstot+=val_loss
            vallosstote+=val_losse
            vallosstotf+=val_lossf
    vallosstot=vallosstot/(stepts+1)
    vallosstote=vallosstote/(stepts+1)
    vallosstotf=vallosstotf/(stepts+1)

    stop_ts=time.time()
    [choice,newbestval]=criterio(vallosstot,vallosstotf,bestval)
    if 1:
       bestval=newbestval
       outfold_name=model_name+str(ep)
       model.save_model(outfold_name)
       np.savetxt(outfold_name+"/model_error",[np.sqrt(vallosstote),np.sqrt(vallosstotf)],header='val_loss_e  val_loss_f ')
    print(np.sqrt(vallosstote.numpy()),np.sqrt(vallosstotf.numpy()),losstot.numpy(),lrnow.numpy(),lrnow2.numpy(),sep=' ',end='\n',file=fileOU)
    stop=time.time()
    print(stop_tr-start,stop_ts-stop_tr,sep=' ',end='\n',file=out_time)
    print(np.sqrt(vallosstote.numpy()),np.sqrt(vallosstotf.numpy()),losstot.numpy(),lrnow.numpy(),lrnow2.numpy(),sep=' ',end='\n')
    print("We are at epoch ",ep)
    fileOU.flush()
    out_time.flush()

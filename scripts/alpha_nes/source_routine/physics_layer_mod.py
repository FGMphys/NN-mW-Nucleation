import tensorflow as tf
import numpy as np

root_path='/home/francegm/programmi/alpha_nes'
proj2b_sopath=root_path+'/bin/projdesonbasisNODEN.so'
proj3b_sopath=root_path+'/bin/proj3bodyonbasis_2alpha1betaNODEN.so'


class physics_layer(tf.Module):
      def __init__(self,init_alpha2b,init_alpha3b,
                   initial_type_emb):
          super(physics_layer, self).__init__()
          self.proj2b=tf.load_op_library(proj2b_sopath)
          self.proj3b=tf.load_op_library(proj3b_sopath)

          self.alpha2b=tf.Variable(init_alpha2b)
          self.alpha3b=tf.Variable(init_alpha3b)
          self.type_emb_2b=tf.Variable(initial_type_emb[0])
          self.type_emb_3b=tf.Variable(initial_type_emb[1])


          self.nalpha_r=init_alpha2b.shape[0]
          self.nalpha_a=init_alpha3b.shape[0]
          self.nt=self.type_emb_2b.shape[0]
          self.output_dim=self.nalpha_r+self.nalpha_a




      @tf.function()
      def train(self,x2b,x3bsupp,nr,intmap2b,x3b,na,intmap3b,N,dimbat,type_map):
          self.type_emb_2b_sq=tf.square(self.type_emb_2b)
          self.type_emb_3b_sq=tf.square(self.type_emb_3b)

          self.resproj2b=self.proj2b.compute_sort_proj(x2b,N,nr,dimbat,intmap2b
                                                       ,self.alpha2b,self.nalpha_r,
                                                       self.type_emb_2b_sq,self.nt,
                                                       type_map)
          ###Da eliminare numneigh2b perchè non viene usato!!!
          self.resproj3b=self.proj3b.compute_sort_proj3body(x3b,x3bsupp,na,nr,
                                                           intmap3b,N,dimbat,
                                                           intmap2b,self.alpha3b,
                                                           self.nalpha_a,
                                                           self.type_emb_3b_sq,
                                                           self.nt,type_map)
          self.restot=tf.concat([self.resproj2b,self.resproj3b],axis=2)
          return self.restot
      @tf.function()
      def test(self,x2b,x3bsupp,nr,intmap2b,x3b,na,intmap3b,N,dimbat,type_map):

          self.type_emb_2b_sq=tf.square(self.type_emb_2b)
          self.type_emb_3b_sq=tf.square(self.type_emb_3b)
          self.resproj2b=self.proj2b.compute_sort_proj(x2b,N,nr,dimbat,intmap2b,
                                                        self.alpha2b,self.nalpha_r,
                                                        self.type_emb_2b_sq,self.nt,type_map)
          self.resproj3b=self.proj3b.compute_sort_proj3body(x3b,x3bsupp,na,nr,
                                                            intmap3b,N,dimbat,
                                                            intmap2b,self.alpha3b,
                                                            self.nalpha_a,
                                                            self.type_emb_3b_sq,
                                                           self.nt,type_map)
          self.restot=tf.concat([self.resproj2b,self.resproj3b],axis=2)
          return self.restot
      def savealphas(self,folder_ou,prefix):
         np.savetxt(folder_ou+'/'+prefix+'alpha_2body.dat',self.alpha2b.numpy())
         np.savetxt(folder_ou+'/'+prefix+'alpha_3body.dat',self.alpha3b.numpy())
         np.savetxt(folder_ou+'/'+prefix+'type_emb_2b.dat',self.type_emb_2b.numpy())
         np.savetxt(folder_ou+'/'+prefix+'type_emb_3b.dat',self.type_emb_3b.numpy())
         np.savetxt(folder_ou+'/'+prefix+'type_emb_2b_sq.dat',self.type_emb_2b.numpy()**2)
         np.savetxt(folder_ou+'/'+prefix+'type_emb_3b_sq.dat',self.type_emb_3b.numpy()**2)
class lognorm_layer(tf.Module):
      def __init__(self,input_mu,mu):
          super(lognorm_layer, self).__init__()
          self.mu=tf.Variable(mu,trainable=True)
      @tf.function()
      def train(self,restot):
          return tf.math.log(restot+10**(-3))-self.mu
      @tf.function()
      def test(self,restot):
          return tf.math.log(restot+10**(-3))-self.mu
      def savemu(self,folder_ou,prefix):
          np.savetxt(folder_ou+'/'+prefix+'alpha_mu.dat',self.mu.numpy())

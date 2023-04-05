import tensorflow as tf

root_path='/home/francegm/programmi/alpha_nes'
force2b_sopath=root_path+'/bin/computeforce_rNODEN.so'
force3b_sopath=root_path+'/bin/computeforce_triplNODEN.so'

class force_layer(tf.Module):
      def __init__(self):
          self.force2b=tf.load_op_library(force2b_sopath)
          self.force3b=tf.load_op_library(force3b_sopath)

      @tf.function()
      def __call__(self,net_der_r,x2b,nr,intder2b,int2b,nalpha_r,alpha2b,net_der_a,
                   x3b,x3bsupp,na,intder3b,intder3bsupp,int3b,nalpha_a,alpha3b,N,
                   dimbat,type_emb_2b,nt,type_emb_3b,type_map):

          self.type_emb_2b_sq=tf.square(type_emb_2b)
          self.type_emb_3b_sq=tf.square(type_emb_3b)

          self.force_radial=self.force2b.compute_force_radial(net_der_r,intder2b,int2b,
                                                              N,nr,dimbat,x2b,nalpha_r,
                                                              alpha2b,self.type_emb_2b_sq
                                                              ,nt,type_map)

          self.force_angular=self.force3b.compute_force_tripl(net_der_a,x3bsupp,x3b,
                                                              intder3bsupp,intder3b,
                                                              int2b,int3b,nr,na,N,dimbat,
                                                              nalpha_a,alpha3b,
                                                              self.type_emb_3b_sq,
                                                              nt,type_map)
          self.forcetot=self.force_radial+self.force_angular
          return self.forcetot

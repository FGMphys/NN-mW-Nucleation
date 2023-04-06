import tensorflow as tf


class alpha_nes_full_inference(tf.Module):
      def __init__(self,physics_layer,force_layer,lognorm_layer,tipos,type_map,modelname):
          super(alpha_nes_full_inference, self).__init__()



          self.physics_layer=physics_layer
          self.lognorm_layer=lognorm_layer
          self.force_layer=force_layer

          self.tipos=tipos
          self.ntipos=len(tipos)
          self.type_map=type_map
          self.N=len(type_map)

          nt=self.physics_layer.nt


          self.nets = [tf.keras.models.load_model(modelname+'/net_model_type'+str(k))
                        for k in range(nt)]


      @tf.function()
      def full_test(self,x,x3bsupp,int2b,intder2b,int3b,intder3b,intder3bsupp):

          nalpha_r=self.physics_layer.nalpha_r
          nalpha_a=self.physics_layer.nalpha_a
          nt=self.ntipos

          nr=int2b.shape[-1]-1
          na=x.shape[-1]-nr
          dimbat=x.shape[0]
          x2b=x[:,:,:nr]
          x3b=x[:,:,nr:]
          N=self.N

          self.restot=self.physics_layer.train(x2b,x3bsupp,nr,int2b,x3b,
                                               na,int3b,N,dimbat,self.type_map)
          self.chemical_proj=tf.split(self.restot,self.tipos,axis=1)
          self.log_norm_projdes=[self.lognorm_layer.train(chemical) for chemical in self.chemical_proj]

          self.energy=[self.nets[k](cp) for k,cp in enumerate(self.log_norm_projdes)]
          self.grad_ene=[tf.gradients(self.energy[k],cp) for k,cp in enumerate(self.chemical_proj)]



          self.totgrad=tf.concat(self.grad_ene,axis=2)
          self.totene=tf.concat(self.energy,axis=1)
          self.totenergy=tf.reduce_mean(self.totene,axis=(-1,-2))

          [self.net_der_r,self.net_der_a]=tf.split(self.totgrad,[nalpha_r,nalpha_a],axis=3)

          self.force=self.force_layer(self.net_der_r,x2b,nr,intder2b,int2b,
                                      nalpha_r,self.physics_layer.alpha2b,
                                      self.net_der_a,x3b,x3bsupp,na,
                                      intder3b,intder3bsupp,int3b,nalpha_a,
                                      self.physics_layer.alpha3b,N,dimbat,
                                      self.physics_layer.type_emb_2b,nt,
                                      self.physics_layer.type_emb_3b,
                                      self.type_map)


          return self.totenergy/2.*self.N,self.force

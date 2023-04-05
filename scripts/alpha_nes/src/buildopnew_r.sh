TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )






g++ -std=c++14 -shared alphagrad_2body.cc -o ../bin/alphagrad_2body.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 


g++ -std=c++14 -shared projdesonbasisNODEN.cc -o ../bin/projdesonbasisNODEN.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 

g++ -std=c++14 -shared computeforce_rNODEN.cc -o ../bin/computeforce_rNODEN.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3

g++ -std=c++14 -shared computeforcegrad_rNODEN.cc -o ../bin/computeforcegrad_rNODEN.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 


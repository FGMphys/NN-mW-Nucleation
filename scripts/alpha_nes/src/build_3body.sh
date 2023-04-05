TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )





g++ -std=c++14 -shared proj3bodyonbasis_2alpha1betaNODEN.cc -o ../bin/proj3bodyonbasis_2alpha1betaNODEN.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm 

g++ -std=c++14 -shared alphagrad_3body.cc -o ../bin/alphagrad_3body.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm 

g++ -std=c++14 -shared computeforce_triplNODEN.cc -o ../bin/computeforce_triplNODEN.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm


g++ -std=c++14 -shared computeforcegrad_triplNODEN_alphagrad2.cc -o ../bin/computeforcegrad_triplNODEN_alpha2.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -lm 

# NN-mW-Nucleation
This repository contains the scripts to run the training of the Neural Network Potential for mW nucleation.
The following python packages are required to run the training program:

1. Tensorflow==2.8 (and its dependency)
2. pyyaml

It may be useful to create a conda environment, then activate and thus run

1. conda install tensorflow==2.8
2. conda install pyyaml

In the scripts/alpha_nes folder there are two files to install the paths (install_path.sh for linux and install_path_mac.sh for mac).
The install scripts should compile also the .cc source files in the src folder.

Once the paths are written correctly and the .cc source files are built, it is possible to run the training by 

"python alpha_nnpes_full_main.py input.yaml"

The file input.yaml is the configuration file to fix the parameters of the training as discussed in the method article (DOI: 10.1063/5.0139245).


The dataset to run the training can be found on Zenodo at DOI: 10.5281/zenodo.7804928.

Once the model is trained, it can be possible to compute energy and force error by running 
"python alpha_nnpes_full_inference_main.py input_inference.yaml" 
The input_inference.yaml is a configuration file for the inference run.



The code development was supported by ICSC—Centro Nazionale di Ricerca in High Performance Computing, Big Data and Quantum Computing, funded by the European Union—NextGenerationEU.

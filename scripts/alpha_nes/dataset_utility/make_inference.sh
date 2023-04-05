
echo "Script to run in configuration folder "

##############INPUT DESCRIPTOR PARAMETERS########
#################################################
programtouse=op_comp_3bcc
rc_radial=1.9
rc_triplets=1.8




number_of_particles=1000
nf=1 #number of frames of the dataset (train+test)
seed=12345 #to select frames
#######################################
##FORCE AND ENERGY DATASET############

position_folder=/home/francegm/Scrivania/CoarseGraining/MW3bodymodel/liquid_mw/simulations/NPTmw_0.07P0/configurations


#######OUTPUT FOLDER###########
name_output_folder_dataset=.







########################NOT MODIFY FROM HERE########

#################################
#### INPUT DATASET PARAMETERS####
################################
dataset_descriptors=descriptor_dataset.csv

dataset_radial_derivative=radialderivat_dataset.csv

dataset_triplets_derivative=tripletsderivat_dataset.csv

dataset_radial_interaction=radialint_dataset.csv

dataset_triplets_interaction=tripletsint_dataset.csv

##Move in the working directory 
back=$(pwd)
path_ut=/home/francegm/Scrivania/CoarseGraining/alpha_nes/dataset_utility
cd $position_folder

rm out_rc_2b.dat
for el in $(ls $(python3 $path_ut'/random_select_name.py' pos_ $nf $seed)| sort -V);
do
cp $el $back
cutoff_neighbours $el $rc_radial >> out_rc_2b.dat
done
num_radial=$(cat out_rc_2b.dat | sort -V | tail -n1 )

rm out_rc_3b.dat
for el in $(ls $(python3 $path_ut'/random_select_name.py' pos_ $nf $seed)| sort -V)
do
cutoff_neighbours $el $rc_triplets >> out_rc_3b.dat
done
temp=$(cat out_rc_3b.dat | sort -V | tail -n1 )
num_triplets=$((temp*(temp-1)/2))

echo Found max 2b and 3b neigh are $num_radial $num_triplets

rm alldes* int2b* int3b* dev2b* dev3b*
for el in $(ls $(python3 $path_ut'/random_select_name.py' pos_ $nf $seed)| sort -V);
do 
$programtouse $el 0 $rc_radial $num_radial $rc_triplets $num_triplets $el; 
echo $el
done;

###Unisco i descrittori 
###Unisco i descrittori
head -n1 $(ls  alldes* | sort -V |head -n1)  > $dataset_descriptors
cat $(ls alldes* | sort -V) | sed  "/\"/d" >> $dataset_descriptors
###Unisco le derivate
head -n1 $(ls dev2b* | sort -V | head -n1) > $dataset_radial_derivative
cat $(ls dev2b* | sort -V) | sed  "/\"/d" >> $dataset_radial_derivative
head -n1 $(ls dev3b* | sort -V | head -n1) > $dataset_triplets_derivative
cat $(ls dev3b* | sort -V) | sed  "/\"/d" >> $dataset_triplets_derivative
###Unisco le interazioni
head -n1 $(ls int2b* | sort -V | head -n1) > $dataset_radial_interaction
cat $(ls int2b* | sort -V )| sed  "/\"/d" >> $dataset_radial_interaction
head -n1 $(ls  int3b* | sort -V | head -n1) > $dataset_triplets_interaction
cat $(ls int3b* | sort -V )| sed  "/\"/d" >> $dataset_triplets_interaction

mv *csv  $back'/'$name_output_folder_dataset
rm alldes* int2b* int3b* dev2b* dev3b*

##Creo dataset delle forze e energie locali
#FORZE
rm dataset_force.dat
echo \"Fx\" \"Fy\" \"Fz\" > dataset_force.dat
for el in $(ls $(python3 $path_ut'/random_select_name.py' force_ $nf $seed)| sort -V);
do
tail -n+2 $el >> dataset_force.dat
echo $el
done;
mv dataset_force.dat $back'/'$name_output_folder_dataset
#POSITION
rm dataset_position.dat
for el in $(ls $(python3 $path_ut'/random_select_name.py' pos_ $nf $seed)| sort -V);
do
paste -s -d " " $el >> dataset_position.dat
echo $el
done;
mv dataset_position.dat $back'/'$name_output_folder_dataset
###PRESSIONE (VIRIALE)
rm dataset_pressure.dat
echo \"Pressure\" > dataset_pressure.dat
for el in $(ls $(python3 $path_ut'/random_select_name.py' pos_ $nf $seed)| sort -V);
do
time=$(awk 'NR==1{print $1}' $el)
echo $time
python3 $path_ut'/findval.py' ../virial.dat $time >> dataset_pressure.dat
done
mv dataset_pressure.dat $back'/'$name_output_folder_dataset
cd $position_folder
#ENERGIE LOCALI
rm dataset_energy.dat
for el in $(ls $(python3 $path_ut'/random_select_name.py' pos_ $nf $seed)| sort -V);
do
echo $el
bond_energy_threshold $el 23.15 2. > 'energy_'$el
done;
echo \"Energy\" > dataset_energy.dat
for el in $(ls energy* | sort -V);
do
awk '{print $2}' $el  >> dataset_energy.dat
echo $el
done;
python3 /home/francegm/Scrivania/CoarseGraining/alpha_nes/dataset_utility/from_single2tot_energy.py dataset_energy.dat $number_of_particles
mv dataset_energy.dat $back'/'$name_output_folder_dataset
rm energy*


cd $back

cd . 

python3 /home/francegm/Scrivania/CoarseGraining/alpha_nes_virial/dataset_utility/make_train_test.py $number_of_particles 

rm $dataset_descriptors $dataset_triplets_interaction  $dataset_radial_derivative $dataset_triplets_derivative $dataset_radial_interaction dataset_energy.dat dataset_force.dat dataset_position.dat


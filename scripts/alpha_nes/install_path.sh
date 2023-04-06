#!/bin/sh
actual_path=$(pwd)

sed -i 's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/physics_layer_mod.py
sed -i 's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/force_layer_mod.py
sed -i 's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/grad_threebody_par.py
sed -i 's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/grad_twobody_par.py
sed -i 's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/gradforceop_rNODEN.py
sed -i 's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/gradforceop_triplNODEN.py
sed -i 's@root_path=.*@root_path='"\'$actual_path\'"'@' dataset_utility/make_train_test_from_inline.py
cd src
bash build_3body.sh
bash buildopnew_r.sh

#!/bin/sh
actual_path=$(pwd)
perl -i -pe's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/physics_layer_mod.py
perl -i -pe's@root_path=.*@root_path='"\'$actual_path\'"'@' source_routine/force_layer_mod.py
perl -i -pe's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/grad_threebody_par.py
perl -i -pe's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/grad_twobody_par.py
perl -i -pe's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/gradforceop_rNODEN.py
perl -i -pe's@root_path=.*@root_path='"\'$actual_path\'"'@' gradient_utility/gradforceop_triplNODEN.py
perl -i -pe's@root_path=.*@root_path='"\'$actual_path\'"'@' dataset_utility/make_train_test_from_inline.py

cd src
bash build_3body.sh
bash buildopnew_r.sh

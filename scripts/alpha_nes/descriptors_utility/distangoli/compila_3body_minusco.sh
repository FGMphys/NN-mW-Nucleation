#!/bin/bash



gcc  -O3 -ffast-math  op_distanze_angoli_minusco.c interaction_map.c cell_list.c events.c log.c secure_search.c utilities.c  -o bin/op_dist_angoli_minusco -lm

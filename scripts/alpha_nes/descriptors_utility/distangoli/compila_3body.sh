#!/bin/bash



gcc  -O3 -ffast-math  op_distanze_angolinewsort.c interaction_map.c cell_list.c events.c log.c secure_search.c utilities.c  -o bin/op_dist_angolinewsort -lm

#!/bin/bash



gcc  -O3 -ffast-math  op_distanze_angoli_window.c interaction_map.c cell_list.c events.c log.c secure_search.c utilities.c  -o op_dist_angoliwindow -lm

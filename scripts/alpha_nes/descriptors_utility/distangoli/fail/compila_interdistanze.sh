#!/bin/bash



gcc -ggdb3 -O3 -ffast-math op_interdistanze_3cutoff.c interaction_map.c cell_list.c events.c log.c secure_search.c utilities.c  -o op_interdistanze -lm

#!/bin/bash



gcc  -O3 -ffast-math  op_3body_cc.c interaction_map.c cell_list.c events.c log.c secure_search.c utilities.c  -o bin/op_comp_3bcc -lm

#!/bin/bash

gcc  -O3 -ffast-math  local_env.c interaction_map.c cell_list.c events.c log.c secure_search.c utilities.c  -o bin/atomic_local_env -lm

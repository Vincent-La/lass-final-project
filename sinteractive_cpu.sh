#!/bin/bash
salloc --partition=class \
       --qos=high \
       --account=class \
       --nodes=1 \
       --ntasks=1 \
       --ntasks-per-node=1 \
       --mem=64gb \
       --time=06:00:00


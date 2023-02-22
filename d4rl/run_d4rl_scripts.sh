#!/bin/bash

~/anaconda3/envs/offrl/bin/python main_off.py --multirun trainer=iql env.name='hopper-medium-v2' device.seed=0,1,2,3 \
device.gpu_idx=0


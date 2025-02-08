#!/bin/sh




###################
##   Parameters  ##
###################
#SUBDIR=offcal-base-differentRS
#listRandomSeed="0 1 2 3 4 5 6 7 8 9 10"
#EARLYSTOP=False
#NumEpochs=15
###################
###################


#SUBDIR=$SUBDIR/

# python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.5

# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=EPSILON --tau=TAU --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=TAU --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=TAU --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &


# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=TAU --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.01 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.02 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.03 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.04 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.05 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.06 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.07 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.08 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.09 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.1 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.11 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.12 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.13 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.14 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.15 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.16 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.17 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.18 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.19 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.2 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.21 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.22 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.23 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.24 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.25 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.26 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.27 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.28 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.29 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.3 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.31 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.32 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.33 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.34 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.35 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.36 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.37 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.38 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.39 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.4 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.41 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.42 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.43 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.44 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.45 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.46 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.47 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.48 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.49 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.5 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.51 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.52 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.53 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.54 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.55 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.56 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.57 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.58 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.59 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.6 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.61 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.62 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.63 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.64 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.65 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.66 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.67 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.68 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.69 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.7 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.71 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.72 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.73 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.74 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.75 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.76 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.77 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.78 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.79 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.8 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.81 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.82 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.83 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.84 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.85 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.86 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.87 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.88 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.89 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.9 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.91 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.92 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.93 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.94 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.96 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.97 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.98 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.99 --result-subdir='folktables_4c_results_DEP_noopt/folktables-kernel/Full_Range' 





exit


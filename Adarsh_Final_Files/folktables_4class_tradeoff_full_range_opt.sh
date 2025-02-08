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


# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=TAU --epsilon=EPSILON --result-subdir='folktables_4c_results_DEP_opt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=TAU --epsilon=EPSILON --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0 --epsilon=0.39 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.01 --epsilon=0.387 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.02 --epsilon=0.384 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.03 --epsilon=0.381 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.04 --epsilon=0.378 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.05 --epsilon=0.375 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.06 --epsilon=0.372 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.07 --epsilon=0.369 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.08 --epsilon=0.366 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.09 --epsilon=0.363 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.1 --epsilon=0.36 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.11 --epsilon=0.356 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.12 --epsilon=0.352 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.13 --epsilon=0.348 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.14 --epsilon=0.344 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.15 --epsilon=0.34 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.16 --epsilon=0.336 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.17 --epsilon=0.332 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.18 --epsilon=0.328 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.19 --epsilon=0.324 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.2 --epsilon=0.32 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.21 --epsilon=0.316 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.22 --epsilon=0.312 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.23 --epsilon=0.308 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.24 --epsilon=0.304 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.25 --epsilon=0.3 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.26 --epsilon=0.296 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.27 --epsilon=0.292 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.28 --epsilon=0.288 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.29 --epsilon=0.284 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.3 --epsilon=0.28 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.31 --epsilon=0.275 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.32 --epsilon=0.27 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.33 --epsilon=0.265 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.34 --epsilon=0.26 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.35 --epsilon=0.255 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.36 --epsilon=0.25 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.37 --epsilon=0.245 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.38 --epsilon=0.24 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.39 --epsilon=0.235 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.4 --epsilon=0.23 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.41 --epsilon=0.224 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.42 --epsilon=0.218 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.43 --epsilon=0.212 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.44 --epsilon=0.206 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.45 --epsilon=0.2 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.46 --epsilon=0.194 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.47 --epsilon=0.188 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.48 --epsilon=0.182 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.49 --epsilon=0.176 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.5 --epsilon=0.17 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.51 --epsilon=0.16525 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.52 --epsilon=0.1605 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.53 --epsilon=0.15575 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.54 --epsilon=0.151 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.55 --epsilon=0.14625 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.56 --epsilon=0.1415 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.57 --epsilon=0.13675 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.58 --epsilon=0.132 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.59 --epsilon=0.12725 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.6 --epsilon=0.1225 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.61 --epsilon=0.11725 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.62 --epsilon=0.112 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.63 --epsilon=0.10675 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.64 --epsilon=0.1015 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.65 --epsilon=0.09625 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.66 --epsilon=0.091 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.67 --epsilon=0.08575 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.68 --epsilon=0.0805 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.69 --epsilon=0.07525 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.7 --epsilon=0.07 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.71 --epsilon=0.0665 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.72 --epsilon=0.063 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.73 --epsilon=0.0595 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.74 --epsilon=0.056 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.75 --epsilon=0.0525 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.76 --epsilon=0.049 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.77 --epsilon=0.0455 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.78 --epsilon=0.042 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.79 --epsilon=0.0385 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.8 --epsilon=0.035 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.81 --epsilon=0.033 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.82 --epsilon=0.031 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.83 --epsilon=0.029 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.84 --epsilon=0.027 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.85 --epsilon=0.025 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.86 --epsilon=0.023 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.87 --epsilon=0.021 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.88 --epsilon=0.019 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.89 --epsilon=0.017 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.9 --epsilon=0.015 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.91 --epsilon=0.013667 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.92 --epsilon=0.012334 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.93 --epsilon=0.011001 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.94 --epsilon=0.009668 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --epsilon=0.008335 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.96 --epsilon=0.00700200000000001 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.97 --epsilon=0.00566900000000001 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.98 --epsilon=0.004336 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.99 --epsilon=0.003003 --result-subdir='folktables_4c_results_DEP_opt_new/folktables-kernel/Full_Range' 


exit


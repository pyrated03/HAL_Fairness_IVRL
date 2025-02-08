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


# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.01 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.02 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.03 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.04 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.05 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.06 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.07 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.08 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.09 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.1 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.11 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.12 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.13 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.14 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.15 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.16 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.17 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.18 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.19 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.2 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.21 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.22 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.23 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.24 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.25 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.26 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.27 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.28 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.29 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.3 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.31 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.32 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.33 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.34 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.35 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.36 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.37 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.38 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.39 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.4 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.41 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.42 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.43 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.44 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.45 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.46 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.47 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.48 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.49 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.5 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.51 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.52 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.53 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.54 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.55 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.56 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.57 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.58 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.59 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.6 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.61 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' & 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.62 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.63 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.64 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.65 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.66 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.67 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.68 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.69 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.7 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.71 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.72 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.73 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.74 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.75 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.76 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.77 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.78 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.79 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.8 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.81 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.82 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.83 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.84 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.85 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.86 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.87 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.88 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.89 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.9 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.91 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.92 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.93 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.94 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.95 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.96 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.97 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.98 --direct-grad=False --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.99 --direct-grad=Faese --result-subdir='folktables_results_noopt/folktables-kernel/Full_Range' 






exit


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



# CUDA_VISIBLE_DEVICES=2 python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.095 &
# CUDA_VISIBLE_DEVICES=2 python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.0975 &
# CUDA_VISIBLE_DEVICES=2 python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.1 &
# CUDA_VISIBLE_DEVICES=2 python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.1025 &
# CUDA_VISIBLE_DEVICES=2 python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.105 

# python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.09 &
# python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.0925 &
# python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.095 &
# python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.0975 &
# python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.1 &
# python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.1025 &
# python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.105 &
# python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.1075 &
# python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.11 

# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.11685 --tau=0.25 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.106 --tau=0.3 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.08715 --tau=0.41 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.06493 --tau=0.57 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0609999999999999 --tau=0.6 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0496600000000008 --tau=0.69 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.04165 --tau=0.75 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.03625 --tau=0.79 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.02858 --tau=0.84 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.017 --tau=0.91 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.000199999999999999 --tau=0.99 --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' 

# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=0.08 --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=0.24 --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=0.26 --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=0.42 --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=0.49 --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=0.66 --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=0.68 --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=0.7 --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=0.82 --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=False --tau=0.92 --result-subdir='celebA_results_noopt/celebA-kernel/Full_Range'

CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.13565 --tau=0.17 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.12119 --tau=0.23 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.11251 --tau=0.27 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0938200000000001 --tau=0.37 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0857 --tau=0.42 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0755500000000001 --tau=0.49 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0741000000000001 --tau=0.5 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0509200000000007 --tau=0.68 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0403 --tau=0.76 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0128 --tau=0.93 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 



exit


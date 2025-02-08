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



# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=EPSILON --tau=TAU --result-subdir='celebA_results_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=EPSILON --tau=TAU --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.182 --tau=0 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.17922 --tau=0.01 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.17644 --tau=0.02 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.17366 --tau=0.03 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.17088 --tau=0.04 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.1681 --tau=0.05 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.16532 --tau=0.06 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.16254 --tau=0.07 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.15976 --tau=0.08 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.15698 --tau=0.09 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.1542 --tau=0.1 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.15155 --tau=0.11 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.1489 --tau=0.12 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.14625 --tau=0.13 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.1436 --tau=0.14 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.14095 --tau=0.15 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.1383 --tau=0.16 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.13565 --tau=0.17 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.133 --tau=0.18 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.13035 --tau=0.19 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.1277 --tau=0.2 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.12553 --tau=0.21 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.12336 --tau=0.22 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.12119 --tau=0.23 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.11902 --tau=0.24 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.11685 --tau=0.25 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.11468 --tau=0.26 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.11251 --tau=0.27 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.11034 --tau=0.28 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.10817 --tau=0.29 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.106 --tau=0.3 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.10426 --tau=0.31 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.10252 --tau=0.32 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.10078 --tau=0.33 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.09904 --tau=0.34 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0973 --tau=0.35 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0955600000000001 --tau=0.36 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0938200000000001 --tau=0.37 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0920800000000001 --tau=0.38 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0903400000000001 --tau=0.39 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0886000000000001 --tau=0.4 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.08715 --tau=0.41 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0857 --tau=0.42 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.08425 --tau=0.43 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0828 --tau=0.44 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.08135 --tau=0.45 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0799 --tau=0.46 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.07845 --tau=0.47 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0770000000000001 --tau=0.48 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0755500000000001 --tau=0.49 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0741000000000001 --tau=0.5 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.07279 --tau=0.51 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.07148 --tau=0.52 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.07017 --tau=0.53 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.06886 --tau=0.54 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.06755 --tau=0.55 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.06624 --tau=0.56 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.06493 --tau=0.57 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.06362 --tau=0.58 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0623099999999999 --tau=0.59 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0609999999999999 --tau=0.6 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.05974 --tau=0.61 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0584800000000001 --tau=0.62 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0572200000000002 --tau=0.63 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0559600000000003 --tau=0.64 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0547000000000004 --tau=0.65 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0534400000000005 --tau=0.66 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0521800000000006 --tau=0.67 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0509200000000007 --tau=0.68 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0496600000000008 --tau=0.69 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0484000000000009 --tau=0.7 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.04705 --tau=0.71 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0457 --tau=0.72 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.04435 --tau=0.73 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.043 --tau=0.74 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.04165 --tau=0.75 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0403 --tau=0.76 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.03895 --tau=0.77 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0376 --tau=0.78 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.03625 --tau=0.79 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0349 --tau=0.8 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.03332 --tau=0.81 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.03174 --tau=0.82 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.03016 --tau=0.83 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.02858 --tau=0.84 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.027 --tau=0.85 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.02542 --tau=0.86 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.02384 --tau=0.87 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.02226 --tau=0.88 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.02068 --tau=0.89 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0191 --tau=0.9 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.017 --tau=0.91 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0149 --tau=0.92 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0128 --tau=0.93 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0107 --tau=0.94 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0086 --tau=0.95 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0065 --tau=0.96 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0044 --tau=0.97 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.0023 --tau=0.98 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True--epsilon=0.000199999999999999 --tau=0.99 --result-subdir='celebA_results_newb_opt/celebA-kernel/Full_Range' 

# CUDA_VISIBLE_DEVICES=0 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=EPSILON --tau=TAU --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &


# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.297 --tau=0 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.28239 --tau=0.01 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.26778 --tau=0.02 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.25317 --tau=0.03 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.23856 --tau=0.04 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.22395 --tau=0.05 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.20934 --tau=0.06 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.19473 --tau=0.07 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.18012 --tau=0.08 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.16551 --tau=0.09 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.1509 --tau=0.1 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.14331 --tau=0.11 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.13572 --tau=0.12 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.12813 --tau=0.13 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range'& 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.12054 --tau=0.14 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.11295 --tau=0.15 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.10536 --tau=0.16 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0977699999999999 --tau=0.17 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0901799999999999 --tau=0.18 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0825899999999999 --tau=0.19 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0749999999999999 --tau=0.2 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.072 --tau=0.21 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0690000000000001 --tau=0.22 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0660000000000002 --tau=0.23 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0630000000000003 --tau=0.24 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0600000000000004 --tau=0.25 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0570000000000005 --tau=0.26 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0540000000000006 --tau=0.27 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0510000000000007 --tau=0.28 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0480000000000008 --tau=0.29 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0450000000000008 --tau=0.3 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0435 --tau=0.31 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0419999999999992 --tau=0.32 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0404999999999984 --tau=0.33 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0389999999999976 --tau=0.34 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0374999999999968 --tau=0.35 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.035999999999996 --tau=0.36 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0344999999999952 --tau=0.37 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0329999999999944 --tau=0.38 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0314999999999935 --tau=0.39 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0299999999999927 --tau=0.4 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0291 --tau=0.41 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0282000000000073 --tau=0.42 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0273000000000146 --tau=0.43 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0264000000000219 --tau=0.44 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0255000000000292 --tau=0.45 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0246000000000365 --tau=0.46 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0237000000000438 --tau=0.47 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0228000000000511 --tau=0.48 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0219000000000584 --tau=0.49 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0210000000000657 --tau=0.5 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0204 --tau=0.51 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
# CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0197999999999343 --tau=0.52 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0191999999998686 --tau=0.53 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0185999999998029 --tau=0.54 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0179999999997372 --tau=0.55 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0173999999996715 --tau=0.56 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0167999999996058 --tau=0.57 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0161999999995401 --tau=0.58 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0155999999994744 --tau=0.59 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0149999999994087 --tau=0.6 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0145 --tau=0.61 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0140000000005913 --tau=0.62 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0135000000011826 --tau=0.63 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0130000000017739 --tau=0.64 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0125000000023652 --tau=0.65 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0120000000029565 --tau=0.66 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0115000000035478 --tau=0.67 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0110000000041391 --tau=0.68 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0105000000047304 --tau=0.69 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0100000000053217 --tau=0.7 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0097 --tau=0.71 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0093999999946783 --tau=0.72 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0090999999893566 --tau=0.73 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0087999999840349 --tau=0.74 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0084999999787132 --tau=0.75 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0081999999733915 --tau=0.76 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0078999999680698 --tau=0.77 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0075999999627481 --tau=0.78 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0072999999574264 --tau=0.79 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0069999999521047 --tau=0.8 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0067 --tau=0.81 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0064000000478953 --tau=0.82 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0061000000957906 --tau=0.83 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0058000001436859 --tau=0.84 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0055000001915812 --tau=0.85 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0052000002394765 --tau=0.86 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.0049000002873718 --tau=0.87 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00460000033526711 --tau=0.88 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00430000038316241 --tau=0.89 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00400000043105771 --tau=0.9 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00372 --tau=0.91 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00343999956894229 --tau=0.92 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00315999913788458 --tau=0.93 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00287999870682687 --tau=0.94 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00259999827576916 --tau=0.95 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00231999784471145 --tau=0.96 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00203999741365374 --tau=0.97 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00175999698259604 --tau=0.98 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main.py --args exps/celeba/args_celeba_kernel.txt --direct-grad=True --epsilon=0.00147999655153833 --tau=0.99 --result-subdir='celebA_results_DEP_fr_opt/celebA-kernel/Full_Range' 



exit


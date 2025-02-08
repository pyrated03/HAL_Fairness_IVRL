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


# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=TAU --epsilon=EPSILON --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0 --epsilon=0.1117 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.01 --epsilon=0.11149 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.02 --epsilon=0.11128 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.03 --epsilon=0.11107 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.04 --epsilon=0.11086 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.05 --epsilon=0.11065 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.06 --epsilon=0.11044 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.07 --epsilon=0.11023 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.08 --epsilon=0.11002 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.09 --epsilon=0.10981 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.1 --epsilon=0.1096 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.11 --epsilon=0.10933 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.12 --epsilon=0.10906 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.13 --epsilon=0.10879 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.14 --epsilon=0.10852 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.15 --epsilon=0.10825 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.16 --epsilon=0.10798 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.17 --epsilon=0.10771 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.18 --epsilon=0.10744 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.19 --epsilon=0.10717 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.2 --epsilon=0.1069 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.21 --epsilon=0.10655 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.22 --epsilon=0.1062 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.23 --epsilon=0.10585 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.24 --epsilon=0.1055 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.25 --epsilon=0.10515 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.26 --epsilon=0.1048 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.27 --epsilon=0.10445 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.28 --epsilon=0.1041 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.29 --epsilon=0.10375 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.3 --epsilon=0.1034 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.31 --epsilon=0.10291 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.32 --epsilon=0.10242 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.33 --epsilon=0.10193 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.34 --epsilon=0.10144 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.35 --epsilon=0.10095 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.36 --epsilon=0.10046 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.37 --epsilon=0.09997 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.38 --epsilon=0.09948 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.39 --epsilon=0.09899 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.4 --epsilon=0.0985 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.41 --epsilon=0.097662 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.42 --epsilon=0.096824 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.43 --epsilon=0.095986 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.44 --epsilon=0.095148 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.45 --epsilon=0.09431 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.46 --epsilon=0.093472 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.47 --epsilon=0.092634 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.48 --epsilon=0.091796 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.49 --epsilon=0.090958 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.5 --epsilon=0.09012 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.51 --epsilon=0.089088 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.52 --epsilon=0.088056 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.53 --epsilon=0.087024 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.54 --epsilon=0.085992 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.55 --epsilon=0.08496 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.56 --epsilon=0.083928 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.57 --epsilon=0.082896 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.58 --epsilon=0.081864 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.59 --epsilon=0.080832 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.6 --epsilon=0.0797999999999999 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.61 --epsilon=0.07797 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.62 --epsilon=0.0761400000000001 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.63 --epsilon=0.0743100000000002 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.64 --epsilon=0.0724800000000003 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.65 --epsilon=0.0706500000000004 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.66 --epsilon=0.0688200000000005 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.67 --epsilon=0.0669900000000006 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.68 --epsilon=0.0651600000000007 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.69 --epsilon=0.0633300000000008 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.7 --epsilon=0.0615000000000009 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.71 --epsilon=0.05897 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.72 --epsilon=0.0564399999999991 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.73 --epsilon=0.0539099999999982 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.74 --epsilon=0.0513799999999973 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.75 --epsilon=0.0488499999999964 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.76 --epsilon=0.0463199999999955 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.77 --epsilon=0.0437899999999946 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.78 --epsilon=0.0412599999999937 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.79 --epsilon=0.0387299999999928 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.8 --epsilon=0.0361999999999919 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.81 --epsilon=0.03388 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.82 --epsilon=0.0315600000000081 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.83 --epsilon=0.0292400000000162 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.84 --epsilon=0.0269200000000243 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.85 --epsilon=0.0246000000000324 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.86 --epsilon=0.0222800000000405 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.87 --epsilon=0.0199600000000486 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.88 --epsilon=0.0176400000000567 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.89 --epsilon=0.0153200000000648 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.9 --epsilon=0.0130000000000729 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.91 --epsilon=0.011789 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.92 --epsilon=0.0105779999999271 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.93 --epsilon=0.0093669999998542 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.94 --epsilon=0.00815599999978131 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.95 --epsilon=0.00694499999970841 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.96 --epsilon=0.00573399999963551 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.97 --epsilon=0.00452299999956261 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.98 --epsilon=0.00331199999948971 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables/args_folk_kernel.txt --tau=0.99 --epsilon=0.0021009999994168 --result-subdir='folktables_results_DEP_opt/folktables-kernel/Full_Range' 

exit


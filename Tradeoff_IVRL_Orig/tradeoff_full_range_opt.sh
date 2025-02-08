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

python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0 --direct-grad=True --epsilon=0.162 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.01 --direct-grad=True --epsilon=0.161 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.02 --direct-grad=True --epsilon=0.16 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.03 --direct-grad=True --epsilon=0.159 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.04 --direct-grad=True --epsilon=0.158 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.05 --direct-grad=True --epsilon=0.157 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.06 --direct-grad=True --epsilon=0.156 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.07 --direct-grad=True --epsilon=0.155 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.08 --direct-grad=True --epsilon=0.154 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.09 --direct-grad=True --epsilon=0.153 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.1 --direct-grad=True --epsilon=0.152 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.11 --direct-grad=True --epsilon=0.151 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.12 --direct-grad=True --epsilon=0.15 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.13 --direct-grad=True --epsilon=0.149 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.14 --direct-grad=True --epsilon=0.148 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.15 --direct-grad=True --epsilon=0.147 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.16 --direct-grad=True --epsilon=0.146 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.17 --direct-grad=True --epsilon=0.145 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.18 --direct-grad=True --epsilon=0.144 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.19 --direct-grad=True --epsilon=0.143 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.2 --direct-grad=True --epsilon=0.142 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.21 --direct-grad=True --epsilon=0.1405 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.22 --direct-grad=True --epsilon=0.139 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.23 --direct-grad=True --epsilon=0.1375 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.24 --direct-grad=True --epsilon=0.136 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.25 --direct-grad=True --epsilon=0.1345 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.26 --direct-grad=True --epsilon=0.133 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.27 --direct-grad=True --epsilon=0.1315 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.28 --direct-grad=True --epsilon=0.13 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.29 --direct-grad=True --epsilon=0.1285 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range'
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.3 --direct-grad=True --epsilon=0.127 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.31 --direct-grad=True --epsilon=0.1255 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.32 --direct-grad=True --epsilon=0.124 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.33 --direct-grad=True --epsilon=0.1225 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.34 --direct-grad=True --epsilon=0.121 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.35 --direct-grad=True --epsilon=0.1195 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.36 --direct-grad=True --epsilon=0.118 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.37 --direct-grad=True --epsilon=0.1165 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.38 --direct-grad=True --epsilon=0.115 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.39 --direct-grad=True --epsilon=0.1135 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range'&
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.4 --direct-grad=True --epsilon=0.112 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.41 --direct-grad=True --epsilon=0.1105 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.42 --direct-grad=True --epsilon=0.109 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.43 --direct-grad=True --epsilon=0.1075 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.44 --direct-grad=True --epsilon=0.106 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.45 --direct-grad=True --epsilon=0.1045 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.46 --direct-grad=True --epsilon=0.103 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.47 --direct-grad=True --epsilon=0.1015 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.48 --direct-grad=True --epsilon=0.1 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.49 --direct-grad=True --epsilon=0.0985 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range'
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --direct-grad=True --epsilon=0.097 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.51 --direct-grad=True --epsilon=0.096 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.52 --direct-grad=True --epsilon=0.095 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.53 --direct-grad=True --epsilon=0.094 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.54 --direct-grad=True --epsilon=0.093 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.55 --direct-grad=True --epsilon=0.092 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.56 --direct-grad=True --epsilon=0.091 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.57 --direct-grad=True --epsilon=0.09 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.58 --direct-grad=True --epsilon=0.089 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.59 --direct-grad=True --epsilon=0.088 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.6 --direct-grad=True --epsilon=0.087 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.61 --direct-grad=True --epsilon=0.085 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.62 --direct-grad=True --epsilon=0.083 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.63 --direct-grad=True --epsilon=0.081 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.64 --direct-grad=True --epsilon=0.079 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.65 --direct-grad=True --epsilon=0.0770000000000001 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.66 --direct-grad=True --epsilon=0.0750000000000001 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.67 --direct-grad=True --epsilon=0.0730000000000001 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.68 --direct-grad=True --epsilon=0.0710000000000001 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.69 --direct-grad=True --epsilon=0.0690000000000001 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.7 --direct-grad=True --epsilon=0.0670000000000001 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.71 --direct-grad=True --epsilon=0.064 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.72 --direct-grad=True --epsilon=0.061 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.73 --direct-grad=True --epsilon=0.0579999999999998 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.74 --direct-grad=True --epsilon=0.0549999999999997 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.75 --direct-grad=True --epsilon=0.0519999999999996 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.76 --direct-grad=True --epsilon=0.0489999999999995 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.77 --direct-grad=True --epsilon=0.0459999999999994 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.78 --direct-grad=True --epsilon=0.0429999999999993 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.79 --direct-grad=True --epsilon=0.0399999999999992 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.8 --direct-grad=True --epsilon=0.0369999999999991 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.81 --direct-grad=True --epsilon=0.036 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.82 --direct-grad=True --epsilon=0.035 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.83 --direct-grad=True --epsilon=0.0340000000000018 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.84 --direct-grad=True --epsilon=0.0330000000000027 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.85 --direct-grad=True --epsilon=0.0320000000000036 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.86 --direct-grad=True --epsilon=0.0310000000000045 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.87 --direct-grad=True --epsilon=0.0300000000000054 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.88 --direct-grad=True --epsilon=0.0290000000000063 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.89 --direct-grad=True --epsilon=0.0280000000000072 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.9 --direct-grad=True --epsilon=0.027000000000008 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.91 --direct-grad=True --epsilon=0.024 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.92 --direct-grad=True --epsilon=0.021 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.93 --direct-grad=True --epsilon=0.017999999999984 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.94 --direct-grad=True --epsilon=0.014999999999976 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.95 --direct-grad=True --epsilon=0.011999999999968 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.96 --direct-grad=True --epsilon=0.00899999999996 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.97 --direct-grad=True --epsilon=0.005999999999952 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.98 --direct-grad=True --epsilon=0.002999999999944 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.99 --direct-grad=True --epsilon=0 --result-subdir='results_opt_new_barrier/Gaussian-Kernel/Full_Range' 

exit


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


# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=DRFF --epsilon=EPSILON --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=2000 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1950 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1900 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1850 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1800 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1750 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1700 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1650 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1600 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1550 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1500 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1450 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1400 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1350 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1300 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1250 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1200 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1150 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1100 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1050 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=1000 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=950 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=900 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=850 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=800 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=750 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=700 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=650 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=600 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=550 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=500 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=450 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=400 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=350 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=300 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=250 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=200 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=150 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=100 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=75 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=50 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=45 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=25 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=3 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=45 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=43 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=40 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=38 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=35 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=33 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=30 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=28 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=25 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=23 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=20 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=18 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=15 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=13 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=10 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=8 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=5 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=4 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=4 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=3 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=False --tau=0.95 --drff=3 --result-subdir='folktables_4c_results_DEP_noopt_DRFF/folktables-kernel/Full_Range' 
exit


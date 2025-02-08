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
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=2000 --epsilon=0.0083 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1950 --epsilon=0.00831 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1900 --epsilon=0.00832 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1850 --epsilon=0.00833 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1800 --epsilon=0.00834 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1750 --epsilon=0.00835 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1700 --epsilon=0.00836 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1650 --epsilon=0.00837 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1600 --epsilon=0.00838 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1550 --epsilon=0.00839 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1500 --epsilon=0.0084 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1450 --epsilon=0.00834 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1400 --epsilon=0.00828 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1350 --epsilon=0.00822 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1300 --epsilon=0.00816 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1250 --epsilon=0.0081 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1200 --epsilon=0.00804 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1150 --epsilon=0.00798 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1100 --epsilon=0.00792000000000001 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1050 --epsilon=0.00786000000000001 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=1000 --epsilon=0.00780000000000001 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=950 --epsilon=0.0088 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=900 --epsilon=0.00979999999999999 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=850 --epsilon=0.0108 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=800 --epsilon=0.0118 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=750 --epsilon=0.0128 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=700 --epsilon=0.0138 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=650 --epsilon=0.0147999999999999 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=600 --epsilon=0.0157999999999999 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=550 --epsilon=0.0167999999999999 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=500 --epsilon=0.0177999999999999 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=450 --epsilon=0.0173 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=400 --epsilon=0.0168000000000001 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=350 --epsilon=0.0163000000000002 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=300 --epsilon=0.0158000000000003 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=250 --epsilon=0.0153000000000004 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=200 --epsilon=0.02028 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=150 --epsilon=0.0252599999999996 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=100 --epsilon=0.0302399999999992 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=75 --epsilon=0.0352199999999988 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=50 --epsilon=0.0401999999999984 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=45 --epsilon=0.0401999999999984 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &

# CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=25 --epsilon=0.0683 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
# CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=3 --epsilon=0.475 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'

CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=45 --epsilon=0.057 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=43 --epsilon=0.066 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=40 --epsilon=0.075 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=38 --epsilon=0.0775 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=35 --epsilon=0.08 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=33 --epsilon=0.0825 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=30 --epsilon=0.085 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=28 --epsilon=0.088 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=25 --epsilon=0.092 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=23 --epsilon=0.096 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=20 --epsilon=0.1 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=18 --epsilon=0.12 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=15 --epsilon=0.11 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=13 --epsilon=0.1 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=10 --epsilon=0.05 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=8 --epsilon=0.2 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=5 --epsilon=0.25 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' &
CUDA_VISIBLE_DEVICES=2 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=4 --epsilon=0.3 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range'
CUDA_VISIBLE_DEVICES=1 python main_withdataloader.py --args exps/folktables-4class/args_folk4c_kernel.txt --direct-grad=True --tau=0.95 --drff=3 --epsilon=0.475 --result-subdir='folktables_4c_results_DEP_opt_DRFF/folktables-kernel/Full_Range' 
exit


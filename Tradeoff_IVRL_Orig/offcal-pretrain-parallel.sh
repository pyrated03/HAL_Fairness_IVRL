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



#CUDA_VISIBLE_DEVICES=2 python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.095 &
#CUDA_VISIBLE_DEVICES=2 python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.0975 &
#CUDA_VISIBLE_DEVICES=2 python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.1 &
#CUDA_VISIBLE_DEVICES=2 python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.1025 &
#CUDA_VISIBLE_DEVICES=2 python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.105 

python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.095 &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.0975 &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.1 &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.1025 &
python main_gaussian.py --args exps/gaussian/args_gaussian_kernel.txt --tau=0.5 --epsilon=0.105 



exit


# config.py

import os
import datetime
import argparse
import json
import configparser
from hal.utils import misc
import re
from ast import literal_eval as make_tuple

def convert_boolian(string):
    if not bool(string): # empty string
        return False
    elif string in ['True', 'true', 'TRUE']:
        return True
    else:
        return False

def parse_args():
    result_path = "results/"
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # result_path = os.path.join(result_path, now)

    parser = argparse.ArgumentParser(description='Controllable Representation Learning')

    # the following two parameters can only be provided at the command line.
    parser.add_argument('--result-path', type=str, default=result_path, metavar='', help='full path to store the results')
    # parser.add_argument('--result-subdir', type=str, default='', metavar='', help='results sub directory')
    
    parser.add_argument("-c", "--config", "--args-file", dest="config_file", default="args.txt", help="Specify a config file", metavar="FILE")
    args, remaining_argv = parser.parse_known_args()

    # ======================= Project Settings =====================================
    parser.add_argument('--project-name', type=str, default='myproject', metavar='', help='name of the project')
    # parser.add_argument('--save-dir', type=str, default=os.path.join(result_path, 'Save'), metavar='', help='save the trained models here')
    # parser.add_argument('--logs-dir', type=str, default=os.path.join(result_path, 'Logs'), metavar='', help='save the training log files here')
    parser.add_argument('--result-subdir', type=str, default='', metavar='', help='results sub directory')
    parser.add_argument('--out-dir', type=str, default=result_path, metavar='', help='save the tensorboard & test output files here')
    parser.add_argument('--results-txt-file', type=str, default='test-log.txt', metavar='', help='save the tensorboard & test output files here')
    parser.add_argument('--monitor', type=json.loads, default={}, metavar='', help='metric based on which we save models')
    parser.add_argument('--checkpoint-max-history', type=int, default=10, metavar='', help='max checkpopint history')
    parser.add_argument('-s', '--save', '--save-results', type=misc.str2bool, dest="save_results",default='No', metavar='', help='save the arguments and the results')

    # ======================= Data Settings =====================================
    parser.add_argument('--dataset-root-test', type=str, default=None, help='path of the data')
    parser.add_argument('--dataset-root-train', type=str, default=None, help='path of the data')
    parser.add_argument('--dataset-test', type=str, default=None, help='name of training dataset')
    parser.add_argument('--dataset-train', type=str, default=None, help='name of training dataset')
    parser.add_argument('--split-test', type=float, default=None, help='test split')
    parser.add_argument('--split-train', type=float, default=None, help='train split')
    parser.add_argument('--test-dev-percent', type=float, default=None, metavar='', help='percentage of dev in test')
    parser.add_argument('--train-dev-percent', type=float, default=None, metavar='', help='percentage of dev in train')    
    parser.add_argument('--resume', type=str, default=None, help='full path of models to resume training')
    parser.add_argument('--nclasses', type=int, default=None, metavar='', dest='noutputs', help='number of classes for classification')
    parser.add_argument('--noutputs', type=int, default=None, metavar='', help='number of outputs, i.e. number of classes for classification')
    parser.add_argument('--input-filename-test', type=str, default=None, help='input test filename for filelist and folderlist')
    parser.add_argument('--label-filename-test', type=str, default=None, help='label test filename for filelist and folderlist')
    parser.add_argument('--input-filename-train', type=str, default=None, help='input train filename for filelist and folderlist')
    parser.add_argument('--label-filename-train', type=str, default=None, help='label train filename for filelist and folderlist')
    parser.add_argument('--loader-input', type=str, default=None, help='input loader')
    parser.add_argument('--loader-label', type=str, default=None, help='label loader')
    parser.add_argument('--dataset-options', type=json.loads, default=None, metavar='', help='additional model-specific parameters')
    parser.add_argument('--transform-trn', type=json.loads, default={}, metavar='', help='training data transforms')
    parser.add_argument('--transform-val', type=json.loads, default={}, metavar='', help='validation data transforms')
    parser.add_argument('--transform-tst', type=json.loads, default={}, metavar='', help='testing data transforms')
    parser.add_argument('--cache-size', type=int, default=None, help='lmdb data loader cache size')
    parser.add_argument('--dataset-type', type=str, default=None, help='dataset type')
    parser.add_argument('--target-attr', type=int, default=None, help='target attribute')
    parser.add_argument('--sensitive-attr', type=str, default=None, help='sensitive attribute')
    parser.add_argument('--features-path', type=str, default=None, help='features directory path', required=False)
    
    # ======================= Network Model Settings ============================
    parser.add_argument('--model-type', type=str, default=None, help='type of network')
    parser.add_argument('--model-pre-type', type=str, default=None, help='type of feature network')
    parser.add_argument('--model-tgt-type', type=str, default=None, help='type of adversarial network')
    parser.add_argument('--model-adv-type', type=str, default=None, help='type of adversarial network')
    parser.add_argument('--model-options', type=json.loads, default={}, metavar='', help='additional model-specific parameters, i.e. \'{"nstack": 1}\'')
    parser.add_argument('--model-options-tgt', type=json.loads, default={}, metavar='', help='additional model-specific parameters, i.e. \'{"nstack": 1}\'')
    parser.add_argument('--model-options-adv', type=json.loads, default={}, metavar='', help='additional model-specific parameters, i.e. \'{"nstack": 1}\'')
    parser.add_argument('--model-pre-options', type=json.loads, default={}, metavar='', help='additional model-specific parameters, i.e. \'{"nstack": 1}\'')
    parser.add_argument('--loss-type', type=str, default=None, help='loss method')
    parser.add_argument('--loss-tgt-type', type=str, default=None, help='loss method')
    parser.add_argument('--adv-loss-type', type=str, default=None, help='Adv loss method')
    parser.add_argument('--loss-options', type=json.loads, default={}, metavar='', help='loss-specific parameters, i.e. \'{"wsigma": 1}\'')
    parser.add_argument('--adv-loss-options', type=json.loads, default={}, metavar='', help='adv loss-specific parameters, i.e. \'{"wsigma": 1}\'')
    parser.add_argument('--evaluation-type', type=str, default=None, help='evaluation method')
    parser.add_argument('--evaluation-options', type=json.loads, default={}, metavar='', help='evaluation-specific parameters, i.e. \'{"topk": 1}\'')
    parser.add_argument('--adv-evaluation-type', type=str, default=None, help='evaluation method')
    parser.add_argument('--adv-evaluation-options', type=json.loads, default={}, metavar='', help='evaluation-specific parameters, i.e. \'{"topk": 1}\'')
    parser.add_argument('--resolution-high', type=int, default=None, help='image resolution height')
    parser.add_argument('--resolution-wide', type=int, default=None, help='image resolution width')
    parser.add_argument('--ndim', type=int, default=None, help='number of feature dimensions')
    parser.add_argument('--r', type=int, default=None, help='number of embedding dimensions')
    parser.add_argument('--nunits', type=int, default=None, help='number of units in hidden layers')
    parser.add_argument('--dropout', type=float, default=None, help='dropout parameter')
    parser.add_argument('--length-scale', type=float, default=None, help='length scale')
    parser.add_argument('--precision', type=int, default=32, help='model precision')

    # ======================= Common Control Settings ================================
    parser.add_argument('--control-type', type=str, default=None, help='control method')
    parser.add_argument('--control-options', type=json.loads, default={'type':None}, metavar='', help='control type options')
    parser.add_argument('--control-criterion', type=str, default=None, help='control criterion type')
    parser.add_argument('--control-criterion-options', type=json.loads, default={}, metavar='', help='control-criterion-specific parameters')
    parser.add_argument('--control-niters', type=int, default=None, help='number of iterations for control model')
    parser.add_argument('--tau', type=float, default=None, help='Tau to trade-off target and control loss')
    parser.add_argument('--lam', type=float, default=None, help='regularization of kernelization')
    parser.add_argument('--fgp-ang', type=int, default=None, help='angle id of lfrkm baseline')
    parser.add_argument('--kernel_reg', type=float, default=None, help='regularization of kernelization')
    parser.add_argument('--noise', type=float, default=None, help='noise level input data')
    parser.add_argument('--control-epoch', type=int, default=None, help='number of iterations for control model')
    parser.add_argument('--control-pretrain', type=convert_boolian, default=False,help='whether to pretrain control for validation')
    parser.add_argument('--fairness-type', type=str, default=None, help='DP or EO or EoO')
    parser.add_argument('--fairness-type1', type=str, default=None, help='DP or EO or EoO')
    parser.add_argument('--fairness-options', type=json.loads, default={}, metavar='', help='fairness metric options')

    # ======================= Control: HSIC Settings ================================
    parser.add_argument('--thresh-type', type=int, default=None, help='conditional or not')
    parser.add_argument('--sigma-x', type=float, default=None, help='sigma of Gaussian kernel')
    parser.add_argument('--sigma-x-ratio', type=float, default=1.0, help='sigma of Gaussian kernel')
    parser.add_argument('--sigma-y', type=float, default=None, help='sigma of Gaussian kernel')
    parser.add_argument('--sigma-y-ratio', type=float, default=1.0, help='sigma of Gaussian kernel')
    parser.add_argument('--sigma-s', type=float, default=None, help='sigma of Gaussian kernel')
    parser.add_argument('--sigma-s-ratio', type=float, default=1.0, help='sigma of Gaussian kernel')
    parser.add_argument('--drff', type=int, default=None, help='Random Fourier Feature Dimension')
    parser.add_argument('--centering', type=misc.str2bool, default=None, help='Mean removal')
    parser.add_argument('--cholesky-factor', type=float, default=None, help='factor of cholesky columns')
    parser.add_argument('--hsic-type', type=str, default=None, help='conditional or not')
    parser.add_argument('--r-adaptive', type=str, default=None, help='adaptive dimensionality')
    parser.add_argument('--pca-energy', type=float, default=None, help='energy thresholding for r')
    parser.add_argument('--kernel-type', type=str, default=None, help='kernel method')
    parser.add_argument('--kernel-type-y', type=str, default=None, help='kernel method')
    parser.add_argument('--kernel-type-s', type=str, default=None, help='kernel method')
    parser.add_argument('--gaussian-sigma', type=str, default=None, help='function finding gaussian sigma')
    parser.add_argument('--kernel-labels', type=str, default=None, help='kernel on labels?')
    parser.add_argument('--kernel-semantic', type=str, default=None, help='kernel on semantic labels?')
    parser.add_argument('--kernel-data', type=str, default=None, help='kernel on input data?')
    parser.add_argument('--kernel-options', type=json.loads, default={}, metavar='', help='kernel-specific parameters')

    # ======================= Control: HGR Settings ================================
    parser.add_argument('--hgrkde-type', type=str, default=None, help='conditional or not')
    parser.add_argument('--control-model-1', type=str, default=None, help='1st control model for HGR')
    parser.add_argument('--control-model-options-1', type=json.loads, default={}, metavar='', help='1st control-model-specific parameters')
    parser.add_argument('--control-model-2', type=str, default=None, help='2nd control model for HGR')
    parser.add_argument('--control-model-options-2', type=json.loads, default={}, metavar='', help='2nd control-model-specific parameters')

    # ======================= Control: ARL Settings ================================
    parser.add_argument('--control-model', type=str, default=None, help='control model')
    parser.add_argument('--control-mode', type=str, default=None, help='control mode')
    parser.add_argument('--control-model-options', type=json.loads, default={}, metavar='', help='control-model-specific parameters')

    # ======================= Training Settings ================================
    parser.add_argument('--ngpu', type=int, default=None, help='number of gpus to use')
    parser.add_argument('--batch-size_test', type=int, default=None, help='batch size for testing')
    parser.add_argument('--batch-size_train', type=int, default=None, help='batch size for training')
    parser.add_argument('--nepochs', type=int, default=None, help='number of epochs to train')
    parser.add_argument('--nepochs-nncc', type=int, default=None, help='number of epochs to train NNCC', required=False)
    parser.add_argument('--num-init-epochs', type=int, default=None, help='(in ARL) number of epochs to train Encoder branch before starting to train Aversarial branch')
    parser.add_argument('--num-adv-train-iters', type=int, default=None, help='(in ARL) number of training iterations for the Adversarial branch in each epochs')
    parser.add_argument('--nepochs-0', type=int, default=None, help='number of epochs to train first phase')
    parser.add_argument('--niters', type=int, default=None, help='number of iterations at test time')
    parser.add_argument('--epoch-number', type=int, default=None, help='epoch number')
    parser.add_argument('--nthreads', type=int, default=None, help='number of threads for data loading')
    parser.add_argument('--manual-seed', type=int, default=None, help='manual seed for randomness')
    parser.add_argument('--check-val-every-n-epochs', type=int, default=1, help='validation every n epochs')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--EarlyStopping', type=convert_boolian, default=False, help='whether to use EarlyStopping')
    parser.add_argument('--earlystop-options', type=json.loads, default={}, metavar='', help='options of EarlyStopping')

    # ======================= Hyperparameter Settings ===========================
    parser.add_argument('--learning-rate', type=float, default=None, help='learning rate')
    parser.add_argument('--optim-method', type=str, default=None, help='the optimization routine ')
    parser.add_argument('--optim-options', type=json.loads, default={}, metavar='', help='optimizer-specific parameters, i.e. \'{"lr": 0.001}\'')
    parser.add_argument('--scheduler-method', type=str, default=None, help='cosine, step, exponential, plateau')
    parser.add_argument('--scheduler-options', type=json.loads, default={}, metavar='', help='optimizer-specific parameters')

    # ======================= Visualizer Settings ===========================
    parser.add_argument('--visualizer', type=str, default='VisualizerTensorboard', help='VisualizerTensorboard or VisualizerVisdom')
    parser.add_argument('--same-env', type=misc.str2bool, default='No', metavar='',help='does not add date and time to the visdom environment name')

    # ======================= Direct Optimizer-Barrier Function ===========================
    parser.add_argument('--direct-grad', type=convert_boolian, default=True, help='Whether to perform constraint optimization using direct gradient or not')
    parser.add_argument('--epsilon', type=float, default=0, help='Constraint: DEP(Z,S) <= epsilon')
    parser.add_argument('--delta', type=float, default=0.001, help='The neighbourhood within which the result is preferred')
    parser.add_argument('--gamma', type=float, default=0.8, help='Slope of Linear part of barrier function')




    if os.path.exists(args.config_file):
        config = configparser.ConfigParser()
        config.read([args.config_file])
        defaults = dict(config.items("Arguments"))
        parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_argv)

    # add date and time to the name of Visdom environment and the result
    if args.visualizer == 'VisualizerVisdom':
        if args.env == '':
            args.env = args.model_type
        if not args.same_env:
            args.env += '_' + now
    # if args.control_type == "HSICReg" or args.control_type == "HSIC":
    #     args.result_path = result_path + str(args.tau) + '_sigma_' + str(args.sigma)
    # else:
    args.result_path = os.path.join(result_path, args.result_subdir)

    # args.logs_dir = now + '_' + str(args.tau) + '_' + args.control_type
    # args.logs_dir = now + '--' + args.control_type + '_' + str(args.lam)
    # args.logs_dir = now + '--' + args.control_type + '_' + 'tau' + '_' + str(args.tau)
    if args.control_type == 'GaussianFGPClassification':
        args.logs_dir = now + '--' + args.control_type + '_' + 'ang' + '_' + str(args.fgp_ang) + '_' + 'seed' + str(
            args.manual_seed)
    else:
        args.logs_dir = now + '--' + args.control_type + '_' + 'tau' + '_' + str(args.tau) + '_' + 'seed' + str(args.manual_seed)
                    # + '_'  + 'sigma-x-ratio' + '_' + str(args.sigma_x_ratio) + '_' + 'sigma-y-ratio' + '_' + str(args.sigma_y_ratio)

    #args._dir = args.result_path
    args.out_dir = os.path.join(args.result_path, args.logs_dir)
    
    try:
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)
    except:
        pass
    
    # args.logs_dir = os.path.join(args.result_path, 'Logs')
    # args.logs_dir = args.result_path

    # refine tuple arguments: this section converts tuples that are
    #                         passed as string back to actual tuples.
    pattern = re.compile('^\(.+\)')

    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        if isinstance(arg_value, str) and pattern.match(arg_value):
            setattr(args, arg_name, make_tuple(arg_value))
            print(arg_name, arg_value)
        elif isinstance(arg_value, dict):
            dict_changed = False
            for key, value in arg_value.items():
                if isinstance(value, str) and pattern.match(value):
                    dict_changed = True
                    arg_value[key] = make_tuple(value)
            if dict_changed:
                setattr(args, arg_name, arg_value)

    return args

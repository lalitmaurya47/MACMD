from torchvision import transforms
from utils import *
from datetime import datetime
#from datasets.dataset import *
import argparse
from loader import *

class setting_config:
    """
    the config of training setting.
    """
    network = 'pvt-b2-macmd' 
    model_config = {
        'num_classes': 9,
        'input_channels': 1,
        
    }


    test_weights = ''

    datasets_name = 'synapse'#'isic2018'

    num_classes = 9
    input_size_h = 224
    input_size_w = 224
    input_channels = 1
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    loss_weight = [0.4, 0.6]
    batch_size = 16
    epochs = 150
    z_spacing = 1
    work_dir = 'results/' + network + '_' + datasets_name + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 1
    save_interval = 50
    threshold = 0.5

    opt = 'AdamW'

    sch = 'CosineAnnealingLR'


    @classmethod
    def set_datasets(cls):
        if cls.datasets_name == 'isic2017':
            cls.data_path = '/users/data/ISIC2017/'
            cls.datasets = isic_loader
            cls.criterion = BceDiceLoss()
        elif cls.datasets_name == 'isic2018':
            cls.data_path =  '/users/data/'# 
            cls.datasets = isic_loader
            cls.criterion = BceDiceLoss()
        elif cls.datasets_name == 'synapse':
            cls.data_path = '/users/data/Synapse/train/'
            cls.datasets = Synapse_dataset
            cls.list_dir = '/users/Synapse/lists_Synapse/'
            cls.volume_path = '/users/data/Synapse/test/'
            cls.model_config['num_classes'] = 9
            cls.criterion = CeDiceLoss(9, cls.loss_weight)
        elif cls.datasets_name == 'acdc':
            cls.data_path = '/users/data/ACDC/'
            cls.datasets = ACDC_dataset
            cls.list_dir = '/users/data/ACDC/lists_ACDC/'
            cls.volume_path = '/users/data/ACDC/'
            cls.model_config['num_classes'] = 4
            cls.criterion = CeDiceLoss(4, cls.loss_weight)
        else:
            raise Exception('datasets_name in not right!')
        print('data path:', cls.data_path)
    

    @classmethod
    def set_opt_sch(cls):
        assert cls.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
        if cls.opt == 'Adadelta':
            cls.lr = 0.01 if not hasattr(cls, 'lr') else cls.lr # default: 1.0 – coefficient that scale delta before it is applied to the parameters
            cls.rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
            cls.eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability 
            cls.weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
        elif cls.opt == 'Adagrad':
            cls.lr = 0.01 if not hasattr(cls, 'lr') else cls.lr # default: 0.01 – learning rate
            cls.lr_decay = 0 # default: 0 – learning rate decay
            cls.eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
            cls.weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
        elif cls.opt == 'Adam':
            cls.lr = 0.001 if not hasattr(cls, 'lr') else cls.lr # default: 1e-3 – learning rate
            cls.betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
            cls.eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability 
            cls.weight_decay = 0.0001 # default: 0 – weight decay (L2 penalty) 
            cls.amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
        elif cls.opt == 'AdamW':
            cls.lr = 0.001 if not hasattr(cls, 'lr') else cls.lr # default: 1e-3 – learning rate
            cls.betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
            cls.eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
            cls.weight_decay = 0.001 # default: 1e-2 – weight decay coefficient
            cls.amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
        elif cls.opt == 'Adamax':
            cls.lr = 2e-3 if not hasattr(cls, 'lr') else cls.lr # default: 2e-3 – learning rate
            cls.betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
            cls.eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
            cls.weight_decay = 0 # default: 0 – weight decay (L2 penalty) 
        elif cls.opt == 'ASGD':
            cls.lr = 0.01 if not hasattr(cls, 'lr') else cls.lr # default: 1e-2 – learning rate 
            cls.lambd = 1e-4 # default: 1e-4 – decay term
            cls.alpha = 0.75 # default: 0.75 – power for eta update
            cls.t0 = 1e6 # default: 1e6 – point at which to start averaging
            cls.weight_decay = 0 # default: 0 – weight decay
        elif cls.opt == 'RMSprop':
            cls.lr = 1e-2 if not hasattr(cls, 'lr') else cls.lr # default: 1e-2 – learning rate
            cls.momentum = 0 # default: 0 – momentum factor
            cls.alpha = 0.99 # default: 0.99 – smoothing constant
            cls.eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
            cls.centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
            cls.weight_decay = 0 # default: 0 – weight decay (L2 penalty)
        elif cls.opt == 'Rprop':
            cls.lr = 1e-2 if not hasattr(cls, 'lr') else cls.lr # default: 1e-2 – learning rate
            cls.etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
            cls.step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes 
        elif cls.opt == 'SGD':
            cls.lr = 0.01 if not hasattr(cls, 'lr') else cls.lr # – learning rate
            cls.momentum = 0.9 # default: 0 – momentum factor 
            cls.weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
            cls.dampening = 0 # default: 0 – dampening for momentum
            cls.nesterov = False # default: False – enables Nesterov momentum 
        
        cls.sch = 'CosineAnnealingLR'
        if cls.sch == 'StepLR':
            cls.step_size = cls.epochs // 5 # – Period of learning rate decay.
            cls.gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
            cls.last_epoch = -1 # – The index of last epoch. Default: -1.
        elif cls.sch == 'MultiStepLR':
            cls.milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
            cls.gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
            cls.last_epoch = -1 # – The index of last epoch. Default: -1.
        elif cls.sch == 'ExponentialLR':
            cls.gamma = 0.99 #  – Multiplicative factor of learning rate decay.
            cls.last_epoch = -1 # – The index of last epoch. Default: -1.
        elif cls.sch == 'CosineAnnealingLR':
            cls.T_max = 50 # – Maximum number of iterations. Cosine function period.
            cls.eta_min = 0.000001 #0.00001 # – Minimum learning rate. Default: 0.
            cls.last_epoch = -1 # – The index of last epoch. Default: -1.  
        elif cls.sch == 'ReduceLROnPlateau':
            cls.mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
            cls.factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
            cls.patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
            cls.threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
            cls.threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
            cls.cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
            cls.min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
            cls.eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
        elif cls.sch == 'CosineAnnealingWarmRestarts':
            cls.T_0 = 50 # – Number of iterations for the first restart.
            cls.T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
            cls.eta_min = 1e-6 # – Minimum learning rate. Default: 0.
            cls.last_epoch = -1 # – The index of last epoch. Default: -1. 
        elif cls.sch == 'WP_MultiStepLR':
            cls.warm_up_epochs = 10
            cls.gamma = 0.1
            cls.milestones = [125, 225]
        elif cls.sch == 'WP_CosineLR':
            cls.warm_up_epochs = 20




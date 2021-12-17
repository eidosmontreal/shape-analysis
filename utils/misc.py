import torch
import torch.nn as nn

import models

import os, json

from typing import Tuple, Type, Dict, NoReturn

def num_parameters(model: Type[nn.Module]) -> int:
    """
    :param model: Torch model 
    :return: Number of parameters in model.
    """
    return sum(p.numel() for p in model.parameters())

def print_dict(args: Dict,fname: str=None) -> NoReturn:
    """
    Prints input dictionary into console.

    :param args: Dictionary
    :fname: File path to save results
    """
    if fname is not None:
        file = open(fname,'w')
    for k,v in zip(args.keys(),args.values()):
        if fname is None:
            print(k + ': ' + str(v))
        else:
            print(k + ': ' + str(v),file=file,flush=True)
    print('\n')

def get_dir_name(args: Dict) -> str:
    """
    Generates file name based on key-value pairs in input dictionary.

    :param: args
    """
    dir_name = ''
    for i,l in enumerate(loss_to_label.keys()):
        s = loss_to_label[l]
        val = '%.1e'%getattr(args,l)
        dir_name += s + '_' + val
        dir_name = dir_name + '_' if i < len(loss_to_label)-1   else dir_name
    return dir_name

def get_model(log_dir: str, load_params: bool=False) -> Tuple[Type[nn.Module],Dict]:
    """
    Returns model used for experiment stored in ``log_dir``.

    :param log_dir: Directory containing experiment logs/weights
    :param load_params: Boolean indicating whether or not to load pretrained parameters
    """
    with open(os.path.join(log_dir,'args.json')) as j:
        args = json.load(j)
    model = getattr(models,args['model'])(args['in_features'],args['out_features'],**args)
    if load_params:
        pretrained_params = torch.load(os.path.join(log_dir,'experiment.pth'))['weights']
        model.load_state_dict(pretrained_params)
    return model, args

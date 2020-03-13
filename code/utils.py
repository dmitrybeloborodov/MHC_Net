import os
from torch import save, load
from collections import OrderedDict

from model import BasicNN

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def convert_state_dict(state_dict):
    '''
    Converts a state dict saved from a dataParallel module to normal
    module state_dict inplace
    
    Parameters:
    ----------
    state_dict: loaded DataParallel model_state
    
    Returns:
    -------
    state_dict
    '''
    if not next(iter(state_dict)).startswith("module."):
        return state_dict  # abort if dict is not a DataParallel model_state
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
return new_state_dict

def save_ensemble_dict(ensemble_name, models, optimizer, metric, epoch):
    '''
    Creates a folder with ensemble_name and saves state_dicts
    of all models in 'nets' list inside
    
    Parameters:
    ----------
    ensemble_name: string, name of ensemble/folder to save to
    models: list of models
    optimizer: optimizer from torch.optim
    metric: value of metric to save (based on mean output)
    epoch: int, current epoch
    
    Returns:
    -------
    None
    '''
    save_folder = os.path.join(
        BASE_DIR,
        '..',
        'models',
        ensemble_name
    )
    for i in range(len(nets)):
        state = {
            'epoch': epoch,
            'model_state': models[i].state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_metric_score': metric
        }
        save_path = os.path.join(
            save_folder,
            'best_model_{:d}.pkl'.format(i)
        )
        save(state, save_path)

def load_ensemble_dict(ensemble_dict):
    ensemble_name = ensemble_dict['name']
    min_n = ensemble_dict['min_neurons']
    max_n = ensemble_dict['max_neurons']
    growth_rate = (max_n - min_n)//ensemble_dict['n_models']
    load_folder = os.path.join(
        BASE_DIR,
        '..',
        'models',
        ensemble_name
    )
    
    models = []
    for i in range(ensemble_dict['n_models']):
        model = BasicNN(n_neurons = min_n + growth_rate*i)
        model_path = os.path.join(
            load_folder,
            'best_model_{:d}.pkl'.format(i)
        )
        state = convert_state_dict(load(model_path)["model_state"])
        model.load_state_dict(state)
        models.append(model)
        
    return models







import sys
import os
import torch
import numpy as np

from utils import convert_state_dict
from data_retriever import make_test_dataset, to_ic50
from model import BasicNN#, NNEnsemble


def test(model_path, test_file_path, is_ensemble=True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    samples, sample_names = make_test_dataset(test_file_path)
    
    if is_ensemble:
        model = NNEnsemble()
        state = convert_state_dict(torch.load(args.model_path)["model_state"])
        model.load_state_dict(state)
    else:
        model = BasicNN()
        state = convert_state_dict(torch.load(args.model_path)["model_state"])
        model.load_state_dict(state)
    model.eval()
    model.to(device)
    
    for sample in zip(samples, sample_names):
        seq = sample[0]
        sample_name = sample[1]
        seq = seq.to(device)
        output = model(seq)
        pred = output.data.cpu().numpy()
        #pred = pred.astype(np.float32)
        print('{},{},{} (affinity),{} (eluted ligand)'.format(sample_name[0], sample_name[1], to_ic50(pred[0]), pred[1]))
    
if __name__ == '__main__':
    model_path = sys.argv[1]
    test_file_path = sys.argv[2]
    test(model_path, test_file_path, is_ensemble=False)
    
    
    
    
    
    
    
    
    
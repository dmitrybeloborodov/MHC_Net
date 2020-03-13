import os
import numpy as np
import pandas as pd
import torch
#from torch import nn
#import torch.nn.functional as F
from tqdm import tqdm
import time
import random

import data_retriever
from model import BasicNN

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BATCH_SIZE = 16

'''
def random_peptides(num, length=9, distribution=None):
    if num == 0:
        return []
    if distribution is None:
        distribution = pd.Series(
            1, index=sorted(data_retriever.COMMON_AMINO_ACIDS))
        distribution /= distribution.sum()

    return [
        ''.join(peptide_sequence)
        for peptide_sequence in
        np.random.choice(
            distribution.index,
            p=distribution.values,
            size=(int(num), int(length)))
    ]
'''

def train(max_epochs=500, add_negatives=True, print_interval=50, val_interval=100):
    # Setup seeds and DataLoader params
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    loader_params = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
    }

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Starting training\n')
    samples = data_retriever.parse_bdata_input(9)
    train_samples, val_samples = data_retriever.split_samples(samples, validate_ratio=0.2)
    #print(train_samples.shape)
    train_dataset = data_retriever.make_train_dataset(train_samples, add_negatives)
    val_dataset = data_retriever.make_train_dataset(val_samples)
    train_generator = torch.utils.data.DataLoader(train_dataset, **loader_params)
    val_generator = torch.utils.data.DataLoader(val_dataset, **loader_params)
    #train_generator = DataGenerator(BATCH_SIZE, train_samples)
    #val_generator = DataGenerator(BATCH_SIZE, val_samples)
    print("Data generated")
    
    model = BasicNN(n_neurons=60)
    model.to(device)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    print("Using SGD optimizer")
    
    loss_fn = nn.MSELoss()
    print("Using MSE loss")
    
    i=0
    best_mse = max_epochs * 10
    while i <= max_epochs:
        #running_loss_train = 0.
        running_loss_val = 0.
        for seqs, labels in train_generator:
            i += 1
            start_ts = time.time()
            model.train()
            seqs = seqs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs)
            
            loss = loss_fn(input=outputs, target=labels)
            loss.backward()
            optimizer.step()
            
            if (i+1) % print_interval == 0:
                fmt_str = "Epoch [{:d}/{:d}], Loss: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    max_epochs,
                    loss.item()
                )
                print(print_str)
            
            if (i+1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for i_val, seqs_val, labels_val in tqdm(enumerate(val_generator)):
                        seqs_val = seqs_val.to(device)
                        labels_val = labels_val.to(device)
                        
                        outputs = model(seqs_val)
                        val_loss = loss_fn(input=outputs, target=labels_val)
                        running_loss_val += val_loss.item()
                print("Epoch {:d}, Running Validation Loss {:.4f}".format(i+1, running_val_loss))
                
                if running_val_loss <= best_mse:
                    state = {
                        "epoch": i+1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_mse": running_val_loss
                    }
                    save_path = os.path.join(
                        BASE_DIR,
                        '..',
                        'models',
                        'best_model.pkl'
                    )
                    torch.save(state, save_path)
                    print('Model saved')
            
if __name__ == '__main__':
    train(max_epochs=500, add_negatives=True, print_interval=50, val_interval=100)
    
    
    
    
    
    
    
    
    
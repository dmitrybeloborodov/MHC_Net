import torch
import torch.nn as nn
#import torch.functional as F

PROTEIN_SEQUENCE_INFO_CHANNELS = 10
LIGAND_SEQUENCE_INFO_CHANNELS = 10


class BasicNN(nn.Module):
    def __init__(self, n_neurons):
        super(BasicNN, self).__init__()
        self.hidden_block = nn.Sequential(
            nn.Linear(21, n_neurons),
            nn.ReLu(inplace=True)
        )
        
        self.out_block = nn.Sequential(
            nn.Linear(n_neurons, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        mutual = self.hidden_block(x)
        out = self.out_block(mutual)
        
        return out
        
'''        
class NNEnsemble(nn.Module):
    def __init__(self, n_models, min_neurons, max_neurons):
        super(NNEnsemble, self).__init__()
        self.n_models = n_models
        #self.base = base
        self.growth_rate = (max_neurons - min_neurons)//n_models
        self.ensemble = []
        for i in range(n_models):
            self.ensemble.append(BasicNN(min_neurons+i*self.growth_rate))
    
    def forward(self, x):
        outs = torch.Tensor(self.n_models, 2, 1)
        for i in range(self.n_models):
            outs[i] = self.ensemble[i](x).data
            
        return torch.mean(...)
'''        
        
        
        
        
        
        
        
        
        
        
        
        
import torch
import torch.nn as nn
import numpy as np

# from einops import rearrange
class CodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(CodingRate, self).__init__()
        self.eps = eps
    
    def forward(self, X):
        #normalize over the dim_heads dimension
        '''
        X with shape (b, h, m, d)
        W with shape (b*h, d, m)
        I with shape (m, m)
        logdet2 with shape (b*h)
        '''
        # X = rearrange(X, 'b h m d -> (b h) m d')
        X = X/torch.norm(X, dim=-1, keepdim=True)
        # print((X @ X.transpose(1,2))[0])
        W = X.transpose(-1,-2)
        
        
        _,_, p, m = W.shape
        scalar = p / (m * self.eps)
        
        product = W.transpose(-1,-2) @ W
        
        # Move to CPU for logdet computation if on MPS (MPS doesn't support logdet)
        if product.device.type == 'mps':
            product_cpu = product.cpu()
            I = torch.eye(m, device='cpu')
            logdet2 = torch.logdet(I + scalar * product_cpu)
            logdet2 = logdet2.to(X.device)  # Move result back to original device
        else:
            I = torch.eye(m, device=W.device)
            logdet2 = torch.logdet(I + scalar * product)
        
        # print(logdet2.shape)
        mcr2s = logdet2.sum(dim=-1)/(2.)
        # print(mcr2s.shape)
        mean_mcr2 = mcr2s.mean()
        stdev = mcr2s.std()
        return (mean_mcr2, stdev)

def cal_sparsity(matrix, is_sparse=False):
    absmatrix = np.abs(matrix)
    #matrix have shape [batch_size, num_patches, dim]
    if is_sparse==True:
        sparsity_list = [np.count_nonzero(absmatrix[i,:,:]==0)/(matrix.shape[1]*matrix.shape[2]) for i in range(matrix.shape[0])]
        sparsity = np.mean(sparsity_list)
        stdev = np.std(sparsity_list)
    else:
        sparsity = None
        stdev = None
    
    return sparsity, stdev
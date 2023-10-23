from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim : int = 4096
    n_layers : int = 32
    n_heads : int = 32
    n_kv_heads : Optional[int] = None
    vocab_size :int = -1 #Will set later
    multiple_of : int = 256
    ffn_dim_multiplier : Optional[float] = None
    norm_eps : float = 1e-5
    
    #KV Caching
    max_batch_size : int = 32
    max_seq_len : int = 2048
    
    device : str = None
    
    
class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        
        #Gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
        
    def norm(self, x: torch.tensor):
        
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
    
    def forward(self, x : torch.tensor):
        
        return self.weight * self.norm(x.float()).type_as(x)
    

def precompute_theta_pos_frequencies(head_dim : int, seq_len : int, device : str, theta : float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    
    theta_numerator = torch.arange(0, head_dim, 2).float()
    
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    
    #M parameter
    m   = torch.arange(seq_len, device=device)
    
    freqs = torch.outer(m, theta).float()
    
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x:torch.Tensor, freqs_complex : torch.Tensor, device:str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1,2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    
    x_rotated = x_complex * freqs_complex

    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    
    return x_out.type_as(x).to(device)


#KV Caching

def repeat_kv(x: torch.Tensor, n_rep :int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep ==1 :
        return x
    return(
        x[:,:,:,None,:]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        
        .reshape(batch_size,seq_len,n_kv_heads * n_rep, head_dim)    
    )
    
class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        #number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        
        #Number of heads for the queries
        self.n_heads_q = args.n_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads # Repetiton of the query
        self.head_dim = args.dim // args.n_heads
        
        
    


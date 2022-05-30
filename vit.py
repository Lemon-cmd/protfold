import torch, torch.nn as nn
import torch.optim as optim, torch.nn.functional as F

from padder import Padder
from einops import rearrange
from einops.layers.torch import Rearrange

class Attention(nn.Module):
    def __init__(self, in_dim, dim_head, heads=8, dropout=0.1):
        super().__init__()
        
        self._heads = heads
        self._q_proj = nn.Linear(in_dim, dim_head * heads)
        self._k_proj = nn.Linear(in_dim, dim_head * heads)
        self._v_proj = nn.Linear(in_dim, dim_head * heads)
        
        self._split = Rearrange('b l (h d) -> (b h) l d', h = heads, d = dim_head)
        
        self._drop = nn.Dropout(dropout)
        
        self._o_proj = nn.Sequential(
            nn.Linear(dim_head * heads, in_dim),
            nn.Dropout(dropout)
        )
        
        
    def _scaled_dot_prod(self, q, k, v):
        shape = q.shape
        q = self._split(q)
        k = self._split(k)
        v = self._split(v)

        # (M x H) x L x L
        a = torch.bmm(q, k.transpose(-1, -2)) / (q.size(-1) ** (0.5))
        a = torch.softmax(a, dim = -1)
        a = self._drop(a)
        
        # v : (M x H) x L x dk
        av = torch.bmm(a, v)
       
        # M x L x (H x dk)
        av = av.reshape(shape)

        # o : M x L x d
        return self._o_proj(av)
                
    def forward(self, x, **kwargs):
        # M x W x (h x d)
        q = self._q_proj(x)
        k = self._k_proj(x)
        v = self._v_proj(x)
        
        o = self._scaled_dot_prod(q, k, v)
        return o   

class PreNorm(nn.Module):
    def __init__(self, in_dim, fn):
        super().__init__()     
        self._fn = fn
        self._norm = nn.LayerNorm(in_dim)
        
    def forward(self, x, **kwargs):
        return self._fn(self._norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        
        self._fn = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, in_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self._fn(x)
    
    
class Transformer(nn.Module):
    def __init__(self, in_dim, dim_head, mlp_dim, depth=1, heads=1, dropout=0.1):
        super().__init__()
        
        self._layers = nn.ModuleList([])
        
        attn_blk = lambda : PreNorm(in_dim, Attention(in_dim, heads = heads, 
                                                      dim_head = dim_head, dropout = dropout))
        
        ffn_blk = lambda : PreNorm(in_dim, FeedForward(in_dim, mlp_dim, dropout = dropout))
        
        for _ in range(depth):
            self._layers.append(nn.ModuleList([attn_blk(), ffn_blk()]))
            
    def forward(self, x):
        for attn, ff in self._layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim, mlp_dim, 
                 dim_head=64, patch_size=1, depth=1, heads=1, pos_len=1000, pool='cls', dropout=0.1, layer_dropout=0.1): 
        
        super().__init__()
        
        assert type(patch_size) in {tuple, int}, 'patch_size must be either an int or a tuple'
        assert pool in {'cls', 'mean'}, 'pool type must be either None, cls (cls token), or mean (mean pooling)'
            
        if type(patch_size) == tuple:
            assert(len(patch_size) == 2)
            d1, d2 = patch_size
        else:
            d1 = d2 = patch_size
                     
        self.pool = pool
        self.padder = Padder(d1, d2)
        self.patch_dim = d1 * d2 * in_channels
    
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = d1, p2 = d2, c = in_channels),                             
            nn.Linear(self.patch_dim, hid_dim)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, hid_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, pos_len + 1, hid_dim))
        
        self.patch_drop = nn.Dropout(dropout)
            
        self.transformer = Transformer(hid_dim, dim_head, mlp_dim, depth, heads, layer_dropout)
                    
        self.mlp_head = nn.Sequential( 
            nn.LayerNorm(hid_dim),
            nn.Linear(hid_dim, out_channels)
        )

    def _pool(self, x):
        return x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        
    def forward(self, x):
        assert(len(x.shape) == 4)
        
        # pad input such that h % p1 = 0 and w % p2 = 0
        x = self.padder(x)

        # turn H x W x C into N x P^2 x C where N = HW / P^2
        x = self.patch_embed(x)

        # add CLS token
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = x.size(0))
        x = torch.cat((x, cls_tokens), dim = 1)
            
        x += self.pos_embed[:, :x.size(1)]
        x = self.patch_drop(x)
            
        # apply transformer
        x = self.transformer(x)
        
        return self.mlp_head(self._pool(x)) 

class ToImage(nn.Module):
    def __init__(self, out_channels, d1, d2):
        super().__init__()
        
        self._d1 = d1
        self._d2 = d2
        self._out_dim = out_channels
        
    def forward(self, x):
        h = w = int(x.size(1) ** 0.5)
        
        if h * w != x.size(1):
            h += 1
        
        x = rearrange(x, 'b (h w) (d1 d2 c) -> b c (h d1) (w d2)', 
                      h = h, w = w, d1 = self._d1, d2 = self._d2, c = self._out_dim)
        return x
    
class MAE(nn.Module):
    def __init__(self, in_channels, out_channels, encoder_dim, decoder_dim,
                 encoder_mlp_dim=128, encoder_depth=1, encoder_heads=8, encoder_dim_head=64, encoder_dropout=0.1,
                 decoder_mlp_dim=128, decoder_depth=1, decoder_heads=8, decoder_dim_head=64, decoder_dropout=0.1,
                 patch_size=16, pos_max_len=1000):
        
        super().__init__()
        assert type(patch_size) in {tuple, int}, 'patch_size must be either an int or a tuple'
        
        if type(patch_size) == tuple:
            assert(len(patch_size) == 2)
            d1, d2 = patch_size
        else:
            d1 = d2 = patch_size
        
        self._p = d1 * d2
        self._padder = Padder(d1, d2)
        in_dim = d1 * d2 * in_channels
        self._p_embed = nn.Parameter(torch.randn(1, pos_max_len, encoder_dim))
        self._patcher = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = d1, p2 = d2, c = in_channels)
        
        self._p_proj = nn.Linear(in_dim, encoder_dim)
        self._encoder = Transformer(encoder_dim, encoder_dim_head, encoder_mlp_dim, encoder_depth, encoder_heads, encoder_dropout)
        
        self._e_proj = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self._decoder = Transformer(decoder_dim, decoder_dim_head, decoder_mlp_dim, decoder_depth, decoder_heads, decoder_dropout)

        self._d_embed = nn.Parameter(torch.randn(1, pos_max_len, decoder_dim))
        self._to_in_dim = nn.Linear(decoder_dim, self._p * out_channels)
        
        self._to_img = ToImage(out_channels, d1, d2)
        
    def forward(self, x):
        _, _, h, w = x.shape
        
        # pad x for patching
        x = self._padder(x)
        
        # patch padded x
        x = self._patcher(x)
        
        # take only what we need
        x = x[:, : (h * w) // self._p]
        
        # project patches
        x = self._p_proj(x)
        x = x + self._p_embed[:, :x.size(1)]
        
        # pass patches to transformer
        x = self._encoder(x)
        
        # project encoder to decoder dim
        x = self._e_proj(x)
        x = self._decoder(x)
        x = x + self._d_embed[:, :x.size(1)]
        
        # project back to original shape
        x = self._to_in_dim(x)
        
        return self._to_img(x)
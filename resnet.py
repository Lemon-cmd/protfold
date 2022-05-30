import torch, torch.nn as nn
import torch.optim as optim, torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

class ReshapeImage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(x.size(1) ** 0.5))
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64, dilation=1, dropout=0.0):
        super().__init__()
                
        self._fn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, dilation=dilation, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=1),
        )
    
        self._drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self._fn(x)
        return self._drop(x)

class ResidualGroup(nn.Module):
    def __init__(self, in_channels, out_channels=64, dropout=0.0):
        super().__init__()
        assert type(dropout) in {float, tuple, list}
        
        if type(dropout) in {tuple, list}:
            assert len(dropout) == 4
        else:
            dropout = [dropout for _ in range(4)]

        dilations = [1, 2, 4, 8]
        self._fn = nn.Sequential(
            *[ResidualBlock(in_channels, out_channels, dilations[i], dropout[i])
             for i in range(4)]
        )

    def forward(self, x):
        return self._fn(x)
    
class ResNet(nn.Module):
    def __init__(self, in_channels, dmat_out_channels, angs_out_channels, num_blks=4, dropout=0.0):
        super().__init__()
        assert type(dropout) in {float, tuple, list}
        
        if type(dropout) in {tuple, list}:
            assert len(dropout) == num_blks           
            if type(dropout[0]) == tuple:
                for plist in dropout:
                    assert len(plist) == 4
            else:
                tmp = []
                for p in dropout:
                    tmp.append([p for _ in range(4)])
                dropout = tmp
        else:
            dropout = [dropout for _ in range(num_blks)]
            
        
        self._blocks = nn.Sequential(
            *[ResidualGroup(in_channels, in_channels, dropout[i]) 
            for i in range(num_blks)]
        )
        
        self._dmat_head = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(in_channels, dmat_out_channels),
            ReshapeImage()
        )
        
        """
        self._angs_head = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            nn.Linear(in_channels, angs_out_channels),
            ReshapeImage()
        )
        """
    
    def forward(self, x):
        embeddings = self._blocks(x)
        return self._dmat_head(embeddings)
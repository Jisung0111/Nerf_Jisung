import torch as th
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, pos_dim, viw_dim, fet_dim = 256):
        super().__init__();
        
        self.pos_dim, self.viw_dim = pos_dim, viw_dim;
        pos_dim, viw_dim = 6 * pos_dim + 3, 6 * viw_dim + 3;
        
        self.mlp0 = nn.Sequential(
            nn.Linear(pos_dim, fet_dim),   nn.ReLU(),
            nn.Linear(fet_dim, fet_dim),  nn.ReLU(),
            nn.Linear(fet_dim, fet_dim),  nn.ReLU(),
            nn.Linear(fet_dim, fet_dim),  nn.ReLU(),
            nn.Linear(fet_dim, fet_dim),  nn.ReLU()
        );

        self.mlp1 = nn.Sequential(
            nn.Linear(fet_dim + pos_dim, fet_dim),    nn.ReLU(),
            nn.Linear(fet_dim, fet_dim),              nn.ReLU(),
            nn.Linear(fet_dim, fet_dim),              nn.ReLU()
        );
        
        self.sig = nn.Linear(fet_dim, 1);
        self.fet = nn.Linear(fet_dim, fet_dim);
        self.viw = nn.Linear(fet_dim + viw_dim, fet_dim // 2);
        self.rgb = nn.Linear(fet_dim // 2 , 3);
    
    def forward(self, pos, viw):
        pos = th.cat([pos] + \
                     [th.sin(x * pos) for x in 2.0 ** th.arange(self.pos_dim)] + \
                     [th.cos(x * pos) for x in 2.0 ** th.arange(self.pos_dim)], dim = -1);
        viw = th.cat([viw] + \
                     [th.sin(x * viw) for x in 2.0 ** th.arange(self.viw_dim)] + \
                     [th.cos(x * viw) for x in 2.0 ** th.arange(self.viw_dim)], dim = -1);
        
        x = self.mlp0(pos.clone());
        x = self.mlp1(th.cat([x, pos], dim = -1));
        
        y = F.relu(self.sig(x));
        x = th.cat([self.fet(x), viw.repeat(1, x.shape[1], 1)], dim = -1);
        x = F.relu(self.viw(x));
        x = th.sigmoid(self.rgb(x));
        
        return y, x;
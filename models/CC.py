import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
from misc import layer
from . import counters
from misc.utils import CrossEntropyLoss2d

import pdb

class CrowdCounter(nn.Module):
    def __init__(self,gpus,model_name):
        super(CrowdCounter, self).__init__()    
        # pdb.set_trace()    
        ccnet =  getattr(getattr(counters, model_name), model_name)

        loc_layer = getattr(layer, 'LocKernelLayer')

        self.CCN = ccnet()
        self.loc = loc_layer()
        if len(gpus)>1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
            self.loc = torch.nn.DataParallel(self.loc, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
            self.loc = self.loc.cuda()
        self.loss_bce_fn = CrossEntropyLoss2d(torch.FloatTensor([1.0,100.0])).cuda()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, img, dot_map):
        density_map = self.CCN(img)
        gt_map = self.loc(dot_map).bool().long()
        # pdb.set_trace()
        self.loss_mse= self.build_loss(density_map, gt_map.squeeze(1))               
        return density_map, gt_map
    
    def build_loss(self, density_map, gt_data):
        loss_mse = self.loss_bce_fn(density_map, gt_data)  
        return loss_mse

    def test_forward(self, img):                               
        density_map = self.CCN(img)                    
        return density_map


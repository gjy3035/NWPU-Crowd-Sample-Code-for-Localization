import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

from models.CC import CrowdCounter
from config import cfg
from misc.utils import *
import pdb
import datasets
from importlib import import_module

class Trainer():
    def __init__(self, cfg_data, pwd):

        self.cfg_data = cfg_data
        self.train_loader, self.val_loader, self.restore_transform = datasets.loading_data(cfg.DATASET)

        self.data_mode = cfg.DATASET
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd

        self.net_name = cfg.NET
        self.net = CrowdCounter(cfg.GPU_ID,self.net_name).cuda()
        self.optimizer = optim.Adam(self.net.CCN.parameters(), lr=cfg.LR, weight_decay=1e-4)
        # self.optimizer = optim.SGD(self.net.parameters(), cfg.LR, momentum=0.95,weight_decay=5e-4)
        self.scheduler = StepLR(self.optimizer, step_size=cfg.NUM_EPOCH_LR_DECAY, gamma=cfg.LR_DECAY)          

        self.train_record = {'best_bce_loss': 1e20, 'best_model_name': ''}
        self.timer = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

        self.epoch = 0
        self.i_tb = 0
        
        if cfg.PRE_GCC:
            self.net.load_state_dict(torch.load(cfg.PRE_GCC_MODEL))

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']

        self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, 'exp', resume=cfg.RESUME)


    def forward(self):

        self.validate()
        for epoch in range(self.epoch,cfg.MAX_EPOCH):
            self.epoch = epoch
                
            # training    
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)

            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )

            # validation
            if epoch%cfg.VAL_FREQ==0 or epoch>cfg.VAL_DENSE_START:
                self.timer['val time'].tic()
                self.validate()
                self.timer['val time'].toc(average=False)
                print( 'val time: {:.2f}s'.format(self.timer['val time'].diff) )

            if epoch > cfg.LR_DECAY_START:
                self.scheduler.step()


    def train(self): # training for all datasets
        self.net.train()
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            img, gt_map = data
            img = Variable(img).cuda()
            gt_map = Variable(gt_map).cuda()

            self.optimizer.zero_grad()
            pred_map, _ = self.net(img, gt_map)
            loss = self.net.loss
            loss.backward()
            self.optimizer.step()

            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.i_tb += 1
                self.writer.add_scalar('train_loss', loss.item(), self.i_tb)
                self.timer['iter time'].toc(average=False)
                print( '[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' % \
                        (self.epoch + 1, i + 1, loss.item(), self.optimizer.param_groups[0]['lr']*10000, self.timer['iter time'].diff) )         


    def validate(self):

        self.net.eval()
        
        losses = AverageMeter()

        for vi, data in enumerate(self.val_loader, 0):
            img, dot_map, attributes_pt = data

            with torch.no_grad():
                img = Variable(img).cuda()
                dot_map = Variable(dot_map).cuda()

                pred_map, loc_gt_map = self.net.forward(img, dot_map)
                pred_map = F.softmax(pred_map,dim=1).data.max(1)
                pred_map = pred_map[1].squeeze_(1)

                # # crop the img and gt_map with a max stride on x and y axis
                # # size: HW: __C_NWPU.TRAIN_SIZE
                # # stack them with a the batchsize: __C_NWPU.TRAIN_BATCH_SIZE
                # crop_imgs, crop_dots, crop_masks = [], [], []
                # b, c, h, w = img.shape
                # rh, rw = self.cfg_data.TRAIN_SIZE
                # for i in range(0, h, rh):
                #     gis, gie = max(min(h-rh, i), 0), min(h, i+rh)
                #     for j in range(0, w, rw):
                #         gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                #         crop_imgs.append(img[:, :, gis:gie, gjs:gje])
                #         crop_dots.append(dot_map[:, :, gis:gie, gjs:gje])
                #         mask = torch.zeros_like(dot_map).cuda()
                #         mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                #         crop_masks.append(mask)
                # crop_imgs, crop_dots, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_dots, crop_masks))

                # # forward may need repeatng
                # crop_preds, crop_loc_gts = [], []
                # nz, bz = crop_imgs.size(0), self.cfg_data.TRAIN_BATCH_SIZE
                # for i in range(0, nz, bz):
                #     gs, gt = i, min(nz, i+bz)
                #     #pdb.set_trace()
                #     # print(crop_imgs[gs:gt].shape)
                #     # print(crop_dots[gs:gt].shape)
                #     crop_pred, crop_loc_gt = self.net.forward(crop_imgs[gs:gt], crop_dots[gs:gt])
                #     crop_pred = F.softmax(crop_pred,dim=1).data.max(1)
                #     crop_pred = crop_pred[1].squeeze_(1)
                #     crop_preds.append(crop_pred)
                #     crop_loc_gts.append(crop_loc_gt)
                # crop_preds = torch.cat(crop_preds, dim=0)
                # crop_loc_gts = torch.cat(crop_loc_gts, dim=0)

                # # splice them to the original size
                # idx = 0
                # pred_map = torch.zeros_like(dot_map).cuda()
                # loc_gt_map = torch.zeros_like(dot_map).cuda()
                # for i in range(0, h, rh):
                #     gis, gie = max(min(h-rh, i), 0), min(h, i+rh)
                #     for j in range(0, w, rw):
                #         gjs, gje = max(min(w-rw, j), 0), min(w, j+rw)
                #         pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                #         loc_gt_map[:, :, gis:gie, gjs:gje] += crop_loc_gts[idx]
                #         idx += 1

                # # for the overlapping area, compute average value
                # mask = crop_masks.sum(dim=0).unsqueeze(0)
                # pred_map = (pred_map / mask).bool().long()
                # loc_gt_map = (loc_gt_map / mask).bool().long()


                pred_map = pred_map.bool().long()
                loc_gt_map = loc_gt_map.bool().long()

                # pdb.set_trace()

                pred_map = pred_map.data.cpu().numpy()
                loc_gt_map = loc_gt_map.data.cpu().numpy()
                



                losses.update(self.net.loss.item())


                if vi==0:
                    vis_results(self.exp_name, self.epoch, self.writer, self.restore_transform, img, pred_map, loc_gt_map)
            
        loss = losses.avg


        self.writer.add_scalar('val_loss', loss, self.epoch + 1)

        self.train_record = update_model(self.net,self.optimizer,self.scheduler,self.epoch,self.i_tb,self.exp_path,self.exp_name, \
            loss,self.train_record,self.log_txt)

        print_NWPU_summary(self.exp_name, self.log_txt,self.epoch,loss,self.train_record)

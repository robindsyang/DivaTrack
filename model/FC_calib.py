from timeit import default_timer as timer
import datetime
import os
import csv
import pandas as pd
import math
from sched import scheduler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import to4x4from9d, trans_err, contact_accuracy, save_csv, fromReftoLocal, kl_div, l2_3d
from einops import rearrange
from tensorboardX import SummaryWriter

torch.autograd.set_detect_anomaly(True)
now = datetime.datetime.now().strftime('%m%d_%H%M%S')
nn.Module.dump_patches = True

joints = ["Chest", "Chest2", "Chest3", "Chest4", "Neck", "Head",
          "RightCollar", "RightShoulder", "RightElbow", "RightWrist",
          "LeftCollar", "LeftShoulder", "LeftElbow", "LeftWrist",
          "RightHip", "RightKnee", "RightAnkle", "RightToe",
          "LeftHip", "LeftKnee", "LeftAnkle", "LeftToe"]

class CalibModel(nn.Module):
    def __init__(self, in_dim, out_dim, emb_dim):
        super(CalibModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, out_dim),
            )
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

    def forward(self, input):
        y = self.layers(input)
        return y

class FC_calib():
    def __init__(self, args, loaders):
        super(FC_calib, self).__init__()
        if args.mode == 'train':
            self.device = torch.device('cuda:'+str(min(args.device, torch.cuda.device_count()-1)) if torch.cuda.is_available() else 'cpu')  
        elif args.mode == 'infer':
            self.device = torch.device('cpu')
        
        self.args = args
        self.loaders = loaders
        self.model = CalibModel(6 * 3 * 9, 22 * 3, 256).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20000, 40000, 60000, 80000], gamma=0.5)      

    def train(self):
        args = self.args
        loaders = self.loaders
        model = self.model.to(self.device)
        optimizer = self.optimizer
      
        summary = None
        
        print('Start training FCCalib')
        for i in range(args.total_iter):
            start_time = timer()
            for _, batch in enumerate(loaders.train):
                model.train()
                local_t, world_t, ref_t, imu_a, contact = batch["local_t"].to(self.device), batch["world_t"].to(self.device),\
                                                          batch["ref_t"].to(self.device), batch["imu_a"].to(self.device),\
                                                          batch["contact"].to(self.device)
                b, l, _, _, _ = local_t.shape
                gt_skel = local_t[:, 0, 1:, :3, 3].reshape(b, -1)
                
                # input
                input = ref_t[:, :, [7, 11, 15], :3, 1:].reshape(b, -1)
                       
                optimizer.zero_grad()
                skel = model(input)
                loss = F.l1_loss(skel, gt_skel)
                loss.backward()
                optimizer.step()                
                
            for _, batch in enumerate(loaders.valid):
                model.eval()
                local_t, world_t, ref_t, imu_a, contact = batch["local_t"].to(self.device), batch["world_t"].to(self.device),\
                                                          batch["ref_t"].to(self.device), batch["imu_a"].to(self.device),\
                                                          batch["contact"].to(self.device)
                b, l, _, _, _ = local_t.shape
                gt_skel = local_t[:, 0, 1:, :3, 3].reshape(b, -1)
                
                # input
                input = ref_t[:, :, [7, 11, 15], :3, 1:].reshape(b, -1)
                       
                skel = model(input)
                v_loss, _ = l2_3d(skel.unsqueeze(1), gt_skel.unsqueeze(1))
                           
            # print out log info
            if (i+1) % args.graph_every == 0:
                print('Iter#: {}'.format(i+1))
                print('T_Lss: {}'.format(loss.item()))
                print('V_Err: {}'.format(v_loss))
                end_time = timer()
                print('{}s/it '.format(end_time - start_time))
                print('-----------------------')
                
            # save model checkpoints
            if (i+1) % args.save_every == 0:
                torch.save(model.state_dict(), args.checkpoint_dir + '/{}_FCCalib_{}.pth'.format(str(now), i+1))
                
            if summary == None:
                summary = SummaryWriter(log_dir=args.tb_dir + '/' + str(now) + '_FCCalib_' + args.tb_comment)
            
            if (i+1) % args.graph_every == 0:
                summary.add_scalar('Skel/Train', loss.item(), i)
                summary.add_scalar('Skel/Valid', v_loss.item(), i)
                
    @torch.no_grad()
    def infer(self):
        args = self.args
        model = CalibModel(6 * 3 * 9, 22 * 3, 256).to(self.device)
        PATH = args.cp_dir + '/{}_FCCalib_{}.pth'.format(args.cp_datetime, args.resume_iter)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        
        OUTPATH = args.result_dir + "/" + args.cp_datetime 
        os.makedirs(OUTPATH, exist_ok=True) 
      
        l = args.window_size
    
        if args.infer_set == 'train':
            loader = self.loaders.train
        elif args.infer_set == 'valid':
            loader = self.loaders.valid
        data_count = loader.dataset.data_count
        
        total_frames = 0
        sum_pe = 0
        
        names = []
        pj_pos_errs = []
        with open(str(OUTPATH) + '/avg_errors.csv','w') as f1:
            writer=csv.writer(f1, delimiter=',', lineterminator='\n')
            # data_count = 1 # test
            for i in range(data_count):              
                idx = i
                sample = loader.dataset.get_item(idx)
                print(sample['name'] + ' inference')
                local_t, world_t, ref_t, imu_a, contact = sample["local_t"].to(self.device), sample["world_t"].to(self.device),\
                                                          sample["ref_t"].to(self.device), sample["imu_a"].to(self.device),\
                                                          sample["contact"].to(self.device)
                b, l, _, _, _ = local_t.shape
                gt_skel = local_t[:, 0, 1:, :3, 3].reshape(b, -1)
                
                # input
                input = ref_t[:, :, [7, 11, 15], :3, 1:].reshape(b, -1)
                       
                skel = model(input)
                avg_pos_err, pos_errs = l2_3d(skel.unsqueeze(1), gt_skel.unsqueeze(1))
                
                sum_pe += torch.sum(pos_errs).item()                
                print("pos_err: {}".format(avg_pos_err.item()))
                writer.writerow([sample['name'], avg_pos_err.item()])
                
                names.append(sample['name'])
                pj_pos_errs.append(pos_errs.squeeze().numpy())
                
                os.makedirs(args.result_dir + "/" + args.cp_datetime, exist_ok=True) 
                save_csv(gt_skel.reshape(-1, 3), OUTPATH + "/FC_Calib_" + str(args.resume_iter//10000) + "_" + sample['name'] + "_gt.csv")
                save_csv(skel.reshape(-1, 3), OUTPATH + "/FC_Calib_" + str(args.resume_iter//10000) + "_" + sample['name'] + "_output.csv")
                
            labels = joints
            pos_errs_pd = pd.DataFrame(pj_pos_errs)
            pos_errs_pd.index = names
            pos_errs_pd.columns = labels
            pos_errs_pd.to_csv(OUTPATH + '/pos_errs.csv')
            print("valid_set total errs | pos: {}".format(sum_pe/(data_count * 22)))
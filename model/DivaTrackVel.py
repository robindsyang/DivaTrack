import datetime
import copy
import os
import csv
import time
import pandas as pd
import math
from tqdm import tqdm
from model.FC_calib import CalibModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils import  l2_3d, spineFK2, to4x4from9d, inference_err, contact_accuracy, save_csv, fromReftoLocal, read_json_as_arg
from einops import rearrange
import numpy as np

torch.autograd.set_detect_anomaly(True)
now = datetime.datetime.now().strftime('%m%d_%H%M%S')
nn.Module.dump_patches = True

def randomMasking(x, p=0.9):
    return torch.dropout(x, p=p, train=True)

def randomFlip(x, p=0.10):
    flipped_x = x.clone()
    rand_tensor = torch.rand(x.shape)
    flip_mask = (rand_tensor < p).type(torch.bool)
    flipped_x[flip_mask] = 1 - flipped_x[flip_mask]
    return flipped_x

def ChangeBasis(num_joints, s_input, s_basis, t_basis):
    # t_basis_inv @ s_basis @ input
    b, l, _ = s_input.shape
    s_basis_mat = s_basis.repeat(1, l, num_joints, 1, 1)
    t_basis_inv_mat = torch.inverse(t_basis).repeat(1, l, num_joints, 1, 1)
    b_input = (t_basis_inv_mat @ (s_basis_mat @ to4x4from9d(num_joints, s_input)))[:, :, :, :3, 1:].reshape(b, l, -1)
    return b_input

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, len: int = 29):
        super().__init__()
        self.len = len
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:self.len].to(x.device)
        return self.dropout(x)

class PoseCModel(nn.Module):
    def __init__(self, window_size, in_dim, pose_dim, emb_dim, hid_dim, lat_dim, num_head, num_layer):
        super(PoseCModel, self).__init__()
        self.kernel_size = window_size - 1
        self.emb = nn.Linear(in_dim, emb_dim)
        self.pos_emb = PositionalEncoding(emb_dim, 0.1, self.kernel_size)
        enc_layer = nn.TransformerEncoderLayer(emb_dim, num_head, hid_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layer)
        
        self.u_decoder = nn.Sequential(
                nn.Conv1d(emb_dim, lat_dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size-1)//2),
                nn.BatchNorm1d(lat_dim),
                nn.Tanh(),
                nn.Conv1d(lat_dim, pose_dim , kernel_size=1, stride=1, padding=0),
                )
        
        self.c_decoder = nn.Sequential(
                nn.Conv1d(emb_dim, lat_dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size-1)//2),
                nn.BatchNorm1d(lat_dim),
                nn.Tanh(),
                nn.Conv1d(lat_dim, 2, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
        )

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

    def forward(self, input):
        x = rearrange(input, 'b l c -> l b c')
        x = self.pos_emb(self.emb(x))
        x = self.encoder(x)
        x = rearrange(x, 'l b c -> b c l')
        u = self.u_decoder(x)
        u = rearrange(u, 'b c l -> b l c')
        c = self.c_decoder(x)
        c = rearrange(c, 'b c l -> b l c')
        return u, c

class UModel(nn.Module):
    def __init__(self, window_size, input_dim, pose_dim, emb_dim, hid_dim, lat_dim, num_head, num_layer):
        super(UModel, self).__init__()
        self.kernel_size = window_size - 1
        self.pose0 = PoseCModel(window_size, input_dim, pose_dim, emb_dim, hid_dim, lat_dim, num_head, num_layer)
        self.pose1 = copy.deepcopy(self.pose0)

    def forward(self, input0, input1):
        pose0, contact0 = self.pose0(input0)
        pose1, contact1 = self.pose1(input1)
        return pose0, pose1, contact0, contact1

class TCVAE_Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, num_head, num_layer):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(emb_dim, num_head, hid_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layer)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
    
    def forward(self, input):
        latent = self.encoder(input)
        mu = latent[0]
        logvar = latent[1]
        return mu, logvar

class TCVAE_Decoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, out_dim, num_head, num_layer):
        super().__init__()
        dec_layer = nn.TransformerEncoderLayer(emb_dim, num_head, hid_dim)
        self.decoder = nn.TransformerEncoder(dec_layer, num_layer)
        self.linear = nn.Linear(emb_dim, out_dim)
        self._reset_parameters()
       
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
    
    def forward(self, input):
        latent = self.decoder(input)
        y = self.linear(latent)
        return y

class TCVAE(nn.Module):
    def __init__(self, window_size, c_dim, in_dim, emb_dim, hid_dim, num_head, num_layer):
        super(TCVAE, self).__init__()
        self.kernel_size = window_size - 1
        self.c_emb = nn.Linear(c_dim, emb_dim)
        self.x_emb = nn.Linear(in_dim, emb_dim)
        self.pos_emb = PositionalEncoding(emb_dim, 0.1, self.kernel_size)
        
        self.encoder = TCVAE_Encoder(emb_dim, hid_dim, num_head, num_layer)
        self.decoder = TCVAE_Decoder(emb_dim, hid_dim, in_dim, num_head, num_layer)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def encode(self, input, condition):
        x = rearrange(input, 'b l c -> l b c')
        c = rearrange(condition, 'b l c -> l b c')
        x = self.pos_emb(self.x_emb(x))
        c = self.pos_emb(self.c_emb(c))
        mu, logvar = self.encoder(torch.cat([x, c], dim=0))
        return mu, logvar
    
    def decode(self, mu, logvar, condition):
        c = rearrange(condition, 'b l c -> l b c')
        c = self.pos_emb(self.c_emb(c))
        l, b, emb_c = c.shape
        z = self.reparametrize(mu, logvar).unsqueeze(0)
        query = self.pos_emb(torch.zeros(l, b, emb_c).to(c.device))
        z = torch.cat((query, c, z), dim=0)
        y = self.decoder(z)
        y = rearrange(y, 'l b c -> b l c')
        return y[:, :l]
    
    def sample(self, z, condition):
        b, l, _ = condition.shape
        c = rearrange(condition, 'b l c -> l b c')
        c = self.pos_emb(self.c_emb(c))
        l, b, emb_c = c.shape
        z = z.unsqueeze(0)
        query = self.pos_emb(torch.zeros(l, b, emb_c).to(c.device))
        z = torch.cat((query, c, z), dim=0)
        y = self.decoder(z)
        y = rearrange(y, 'l b c -> b l c')
        return y[:, :l]

    def forward(self, input, condition):
        mu, logvar = self.encode(input, condition)
        y = self.decode(mu, logvar, condition)
        return mu, logvar, y

class LModel(nn.Module):
    def __init__(self, window_size, c_dim, in_dim, emb_dim, hid_dim, num_head, num_layer):
        super(LModel, self).__init__()
        self.pose0 = TCVAE(window_size, c_dim, in_dim, emb_dim, hid_dim, num_head, num_layer)
        self.pose1 = copy.deepcopy(self.pose0)

    def sample(self, z, condition0, condition1):
        pose0 = self.pose0.sample(z, condition0)
        pose1 = self.pose1.sample(z, condition1)
        return pose0, pose1

    def forward(self, input0, condition0, input1, condition1):
        mu0, logvar0, pose0 = self.pose0(input0, condition0)
        mu1, logvar1, pose1 = self.pose1(input1, condition1)
        return mu0, logvar0, pose0, mu1, logvar1, pose1

class BlendModel(nn.Module):
    def __init__(self, window_size, in_dim, pose_dim, emb_dim, lat_dim, hid_dim, num_head, num_layer):
        super(BlendModel, self).__init__()
        self.kernel_size = window_size - 1
        self.emb_in = nn.Linear(in_dim, emb_dim)
        self.emb_pose0 = nn.Linear(pose_dim + 2, emb_dim)
        self.emb_pose1 = nn.Linear(pose_dim + 2, emb_dim)
        self.pos_emb = PositionalEncoding(emb_dim, 0.1, 29)
        enc_layer = nn.TransformerEncoderLayer(emb_dim, num_head, hid_dim)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layer)

        self.f_decoder = nn.Sequential(
                nn.Conv1d(emb_dim, lat_dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size-1)//2),
                nn.BatchNorm1d(lat_dim),
                nn.Tanh(),
                nn.Conv1d(lat_dim, pose_dim , kernel_size=1, stride=1, padding=0),
                )
        
        self.c_decoder = nn.Sequential(
                nn.Conv1d(emb_dim, lat_dim, kernel_size=self.kernel_size, stride=1, padding=(self.kernel_size-1)//2),
                nn.BatchNorm1d(lat_dim),
                nn.Tanh(),
                nn.Conv1d(lat_dim, 2, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
        )
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

    def forward(self, input, pose0, pose1, contact0, contact1):
        pose0 = torch.cat((pose0, contact0), dim=-1)
        pose1 = torch.cat((pose1, contact1), dim=-1)
        x_i = rearrange(input, 'b l c -> l b c')
        x_p0 = rearrange(pose0, 'b l c -> l b c')
        x_p1 = rearrange(pose1, 'b l c -> l b c')
        l = pose0.shape[1]
        x_i = self.pos_emb(self.emb_in(x_i))
        x_p0 = self.pos_emb(self.emb_pose0(x_p0))
        x_p1 = self.pos_emb(self.emb_pose1(x_p1))
        x = torch.cat((x_p0, x_p1, x_i), dim=0)
        x = self.encoder(x)
        x = rearrange(x, 'l b c -> b c l')
        x = x[:, :, :l]
        f = self.f_decoder(x)
        c = self.c_decoder(x)
        f = rearrange(f, 'b c l -> b l c')
        c = rearrange(c, 'b c l -> b l c')
        return f, c
    
class DivaTrackVel():
    def __init__(self, args, loaders):
        super(DivaTrackVel, self).__init__()
        if args.mode == 'train':
            self.device = torch.device('cuda:'+str(min(args.device, torch.cuda.device_count()-1)) if torch.cuda.is_available() else 'cpu')  
        elif args.mode == 'infer' or args.mode == 'export':
            self.device = torch.device('cpu')
                
        self.args = args
        self.loaders = loaders
        self.umodel = UModel(args.window_size, (4 * 9) + (4 * 9) + (3 * 3), (15 * 9), args.emb_dim, args.hid_dim, args.lat_dim, args.n_head, args.n_layer).to(self.device)
        self.uoptimizer = torch.optim.Adam(self.umodel.parameters(), lr=args.lr)
        self.lmodel = LModel(args.window_size, (15 * 9) + (8 * 9) + (2 * 1), (8 * 9), args.emb_dim, args.hid_dim, args.n_head, args.n_layer).to(self.device)
        self.loptimizer = torch.optim.Adam(self.lmodel.parameters(), lr=args.lr)
        self.bmodel = BlendModel(args.window_size, (4 * 9) + (4 * 9) + (3 * 3), (23 * 9), args.emb_dim, args.hid_dim, args.lat_dim, args.n_head, args.n_layer).to(self.device)
        self.boptimizer = torch.optim.Adam(self.bmodel.parameters(), lr=args.lr)
    
    @torch.no_grad()
    def infer(self):
        print("using {} for inference...".format(self.device))
        in_args = self.args
        cp_datetime = in_args.cp_datetime
        resume_iter = in_args.resume_iter

        dir_name = in_args.cp_dir + '/{}_{}/'.format(cp_datetime, in_args.model)
        saved_args = read_json_as_arg(dir_name + 'args.json'.format(cp_datetime))

        # use calib model after training it with FC_calib.py
        # calibmodel = CalibModel(6 * 3 * 9, 22 * 3, 256).to(self.device)
        # cPATH = in_args.cp_dir + '/0113_210014_FCCalib_6000.pth'
        # calibmodel.load_state_dict(torch.load(cPATH))
        # calibmodel.eval()
        # calibloader = self.loaders.skel

        umodel = UModel(saved_args.window_size, (4 * 9) + (4 * 9) + (3 * 3), (15 * 9), saved_args.emb_dim, saved_args.hid_dim, saved_args.lat_dim, saved_args.n_head, saved_args.n_layer)
        lmodel = LModel(saved_args.window_size, (15 * 9) + (8 * 9) + 2, (8 * 9), saved_args.emb_dim, saved_args.hid_dim, saved_args.n_head, saved_args.n_layer)
        bmodel = BlendModel(saved_args.window_size, (4 * 9) + (4 * 9) + (3 * 3), (23 * 9), saved_args.emb_dim, saved_args.hid_dim, saved_args.lat_dim, saved_args.n_head, saved_args.n_layer)

        uPATH = dir_name + 'TCVAE_U_{}.pth'.format(resume_iter)
        lPATH = dir_name + 'TCVAE_L_{}.pth'.format(resume_iter)
        bPATH = dir_name + 'TCVAE_B_{}.pth'.format(resume_iter)

        OUTPATH = in_args.result_dir + "/" + cp_datetime + "_" + str(resume_iter) + "_" + in_args.tb_comment

        umodel.load_state_dict(torch.load(uPATH, map_location=self.device))
        lmodel.load_state_dict(torch.load(lPATH, map_location=self.device))
        bmodel.load_state_dict(torch.load(bPATH, map_location=self.device))
        umodel.eval()
        lmodel.eval()
        bmodel.eval()

        os.makedirs(OUTPATH, exist_ok=True) 
      
        l = saved_args.window_size

        if in_args.infer_set == 'train':
            loader = self.loaders.train
        elif in_args.infer_set == 'valid':
            loader = self.loaders.valid
        data_count = loader.dataset.data_count
        
        total_frames = 0
        names = []
        
        errs = []
        errs_h = []
        errs_t = []
        
        sum_errs = []
        sum_errs_h = []
        sum_errs_t = []

        pj_pos_errs = []
        pj_pos_errs_h = []
        pj_pos_errs_t = []

        pj_rot_errs = []
        pj_rot_errs_h = []
        pj_rot_errs_t = []

        pj_linvel_errs = []
        pj_linvel_errs_h = []
        pj_linvel_errs_t = []

        pj_angvel_errs = []
        pj_angvel_errs_h = []
        pj_angvel_errs_t = []

        target_anims = ["010", "014", "016", "017", "018", "022", "029", "030", "032", "033", "034", "035", "043"]

        for i in range(data_count):              
            idx = i
            sample = loader.dataset.get_item(idx)

            run = False
            if in_args.partial == "True":
                for action in target_anims:
                    if action in sample['name']:
                        run = True
            else:
                run = True

            if run == False:
                continue

            print(str(idx) + ": " + sample['name'] + " inference")
            local_t_full, world_t_full, ref_t_full, imu_a_full, contact_full \
                                    = sample["local_t"].to(self.device), sample["world_t"].to(self.device),\
                                    sample["ref_t"].to(self.device), sample["imu_a"].to(self.device),\
                                    sample["contact"].to(self.device)
            
            # if "totalcapture" and "hps" not in in_args.valid_set:
            #     # calib start
            #     csample = calibloader.dataset.get_item_by_name(sample['name'])
            #     print(csample['name'] + ' calibration')
            #     caliblocal_t, calibworld_t, calibref_t, calibimu_a, calibcontact = csample["local_t"].to(self.device), csample["world_t"].to(self.device),\
            #                                                 csample["ref_t"].to(self.device), csample["imu_a"].to(self.device),\
            #                                                 csample["contact"].to(self.device)
            #     gtcalib_skel = caliblocal_t[:, 0, 1:, :3, 3].reshape(1, 66)
            #     calibinput = calibref_t[:, :, [7, 11, 15], :3, 1:].reshape(1, -1)    
            #     predcalib_skel = calibmodel(calibinput)
            #     calib_avg_pos_err, per_joint_calib_avg_pos_err = l2_3d(predcalib_skel.unsqueeze(1), gtcalib_skel.unsqueeze(1))
            #     print("skel err: {}".format(calib_avg_pos_err.item()))
            #     predcalib_skel = predcalib_skel.reshape(1, 1, 66).repeat(1, 29, 1)

            b, frames, _, _, _ = local_t_full.shape
            
            gt_inputs = torch.zeros(frames-(l+1), 45).to(self.device)

            gt_origin = torch.zeros(frames-(l+1), 9).to(self.device)
            gt_pose_h = torch.zeros(frames-(l+1), 207).to(self.device)   
            gt_pose_t = torch.zeros(frames-(l+1), 207).to(self.device)   
            gt_pose_local_h = torch.zeros(frames-(l+1), 207).to(self.device)
            gt_pose_local_t = torch.zeros(frames-(l+1), 207).to(self.device)
            gt_contact = torch.zeros(frames-(l+1), 2).to(self.device)
            
            pred_origin = torch.zeros(frames-(l+1), 9).to(self.device)
            pred_pose = torch.zeros(frames-(l+1), 207).to(self.device)
            pred_pose_local = torch.zeros(frames-(l+1), 207).to(self.device)
            pred_contact = torch.zeros(frames-(l+1), 2).to(self.device)
            
            pred_origin_h = torch.zeros(frames-(l+1), 9).to(self.device)
            pred_pose_h = torch.zeros(frames-(l+1), 207).to(self.device)
            pred_pose_local_h = torch.zeros(frames-(l+1), 207).to(self.device)
            pred_contact_h = torch.zeros(frames-(l+1), 2).to(self.device)
            
            pred_origin_t = torch.zeros(frames-(l+1), 9).to(self.device)
            pred_pose_t = torch.zeros(frames-(l+1), 207).to(self.device)
            pred_pose_local_t = torch.zeros(frames-(l+1), 207).to(self.device)
            pred_contact_t = torch.zeros(frames-(l+1), 2).to(self.device)
            
            prev_lpose_h = torch.zeros(b, l-1, 72).to(self.device)
            prev_lpose_t = torch.zeros(b, l-1, 72).to(self.device)
            
            for f in tqdm(range(l + 1, frames)):
                local_t = local_t_full[:, f-l:f]
                world_t = world_t_full[:, f-l:f]
                ref_t = ref_t_full[:, f-l:f]
                imu_a = imu_a_full[:, f-l:f]
                contact = contact_full[:, f-l:f]
                gt_contact_window = contact[:, 1:].reshape(b, l-1, -1)
                
                pred_skel = local_t[:, 1:, :, :3, 3].reshape(b, l-1, -1)
                # if in_args.provide_gtskel == "False":
                #     pred_skel[:, :, 3:] = predcalib_skel

                gt_skel = local_t[:, 1:, :, :3, 3].reshape(b, l-1, -1)
                
                ref_pos = ref_t[:, 1:, 0, :3, 3]           
                ref_trans = ref_t[:, 1:, 0]
                ref_vel = ref_t[:, 1:, 0] @ torch.inverse(ref_t[:, :-1, 0])
                
                world_3p_trans = world_t[:, 1:, [6, 10, 14]]
                world_3p_vel = world_t[:, 1:, [6, 10, 14]] @ torch.inverse(world_t[:, :-1, [6, 10, 14]])
                imu_a = imu_a[:, 1:].unsqueeze(-1)
                    
                origin_idx = (l-1)//2

                # origin by hmd
                origin_mat_hmd = ref_trans[:, origin_idx].unsqueeze(1).unsqueeze(1)
                origin_mat_inv_1_hmd = torch.inverse(origin_mat_hmd).repeat(1, l-1, 1, 1, 1)
                origin_mat_inv_3_hmd = torch.inverse(origin_mat_hmd).repeat(1, l-1, 3, 1, 1)
                origin_mat_inv_23_hmd = torch.inverse(origin_mat_hmd).repeat(1, l, 23, 1, 1)

                # input w.r.t. hmd
                h_origin_ref_trans = (origin_mat_inv_1_hmd @ ref_trans.unsqueeze(2))[:, :, :, :3, 1:].reshape(b, l-1, -1)
                h_origin_ref_vel = (origin_mat_inv_1_hmd @ ref_vel.unsqueeze(2))[:, :, :, :3, 1:].reshape(b, l-1, -1)
                h_origin_3p_trans = (origin_mat_inv_3_hmd[:, :, :] @ world_3p_trans[:, :, :])[:, :, :, :3, 1:].reshape(b, l-1, -1)
                h_origin_3p_vel= (origin_mat_inv_3_hmd[:, :, :] @ world_3p_vel[:, :, :])   [:, :, :, :3, 1:].reshape(b, l-1, -1)
                h_origin_rot_inv = origin_mat_inv_3_hmd[:, :, :, :3, :3]
                h_origin_imu_a = (h_origin_rot_inv[:, :, :] @ imu_a[:, :, :]).reshape(b, l-1, -1)
                uinput_h = torch.cat((h_origin_ref_trans, h_origin_ref_vel, h_origin_3p_trans, h_origin_3p_vel, h_origin_imu_a), dim=-1) 
                                
                # origin by trajectory
                origin_pos = ref_pos[:, origin_idx].unsqueeze(-1)
                origin_fwd = (ref_pos[:, -1] - ref_pos[:, 0]).unsqueeze(-1)
                origin_fwd = origin_fwd / torch.norm(origin_fwd, dim=1, keepdim=True)
                origin_up = torch.tensor([0, 1, 0]).float().repeat(b, 1).to(self.device).unsqueeze(-1)
                origin_9d = torch.cat((origin_up, origin_fwd, origin_pos), dim=-1).reshape(b, 1, -1)
                origin_mat_traj = to4x4from9d(1, origin_9d)
                origin_mat_inv_1_traj = torch.inverse(origin_mat_traj).repeat(1, l-1, 1, 1, 1)
                origin_mat_inv_3_traj = torch.inverse(origin_mat_traj).repeat(1, l-1, 3, 1, 1)
                origin_mat_inv_23_traj = torch.inverse(origin_mat_traj).repeat(1, l, 23, 1, 1)
                
                # input by trajectory
                t_origin_ref_trans = (origin_mat_inv_1_traj @ ref_trans.unsqueeze(2))[:, :, :, :3, 1:].reshape(b, l-1, -1)
                t_origin_ref_vel = (origin_mat_inv_1_traj @ ref_vel.unsqueeze(2))[:, :, :, :3, 1:].reshape(b, l-1, -1)
                t_origin_3p_trans = (origin_mat_inv_3_traj[:, :, :] @ world_3p_trans[:, :, :])[:, :, :, :3, 1:].reshape(b, l-1, -1)
                t_origin_3p_vel= (origin_mat_inv_3_traj[:, :, :] @ world_3p_vel[:, :, :])   [:, :, :, :3, 1:].reshape(b, l-1, -1)
                t_origin_rot_inv = origin_mat_inv_3_traj[:, :, :, :3, :3]
                t_origin_imu_a = (t_origin_rot_inv[:, :, :] @ imu_a[:, :, :]).reshape(b, l-1, -1)
                uinput_t = torch.cat((t_origin_ref_trans, t_origin_ref_vel, t_origin_3p_trans, t_origin_3p_vel, t_origin_imu_a), dim=-1) 
                
                # gt for loss
                gt_local_hmd = (origin_mat_inv_23_hmd[:, :, :] @ world_t)[:, :, :, :3, 1:].reshape(b, l, -1)                                
                gt_local_traj = (origin_mat_inv_23_traj[:, :, :] @ world_t)[:, :, :, :3, 1:].reshape(b, l, -1)                                
                upose_h, upose_t, contact_h, contact_t = umodel(uinput_h, uinput_t)
                
                if in_args.provide_gtcont == "True":
                    condition_h = torch.cat((upose_h, prev_lpose_h, gt_contact_window), dim=-1)
                    condition_t = torch.cat((upose_t, prev_lpose_t, gt_contact_window), dim=-1)    
                else:
                    condition_h = torch.cat((upose_h, prev_lpose_h, contact_h), dim=-1)
                    condition_t = torch.cat((upose_t, prev_lpose_t, contact_t), dim=-1)                   
                z = torch.randn(saved_args.emb_dim).unsqueeze(0).to(condition_h.device)
                lpose_h, lpose_t = lmodel.sample(z, condition_h, condition_t)
                prev_lpose_h = torch.cat((prev_lpose_h[:, 1:], lpose_h[:, -1].unsqueeze(1)), dim=1)
                prev_lpose_t = torch.cat((prev_lpose_t[:, 1:], lpose_t[:, -1].unsqueeze(1)), dim=1)
                
                pose_h = torch.cat((upose_h, lpose_h), dim=-1)
                pose_t = torch.cat((upose_t, lpose_t), dim=-1)
                pose_th = ChangeBasis(23, pose_t, origin_mat_traj, origin_mat_hmd)
                pose_hb, contact_hb = bmodel(uinput_h, pose_h, pose_th, contact_h, contact_t) 
                pose_h_gt = gt_local_hmd[:, 1:, :]
                pose_t_gt = gt_local_traj[:, 1:, :]           
                
                gt_input = torch.cat((origin_mat_hmd[:, :, :, :3, 1:].reshape(1, -1), h_origin_3p_trans[:, -1], h_origin_imu_a[:, -1]), dim=-1)
                gt_inputs[f-(l+1)] = gt_input.reshape(-1)
                                                            
                gt_origin[f-(l+1)] = origin_mat_hmd[:, :, :, :3, 1:].reshape(-1)
                gt_fullpose_local_hmd = fromReftoLocal(23, pose_h_gt, gt_skel)
                gt_pelvis_local_hmd = spineFK2(h_origin_3p_trans[:, :, :9], gt_fullpose_local_hmd)
                gt_fullpose_local_hmd[:, :, 0] = gt_pelvis_local_hmd[:, :, 0]
                gt_fullpose_local_hmd = gt_fullpose_local_hmd[:, :, :, :3, 1:]
                gt_fullpose_local_traj = fromReftoLocal(23, pose_t_gt, gt_skel)
                gt_pelvis_local_traj = spineFK2(t_origin_3p_trans[:, :, :9], gt_fullpose_local_traj)
                gt_fullpose_local_traj[:, :, 0] = gt_pelvis_local_traj[:, :, 0]
                gt_fullpose_local_traj = gt_fullpose_local_traj[:, :, :, :3, 1:]
                gt_pose_h[f-(l+1)] = pose_h_gt[:, -1].reshape(-1)
                gt_pose_t[f-(l+1)] = pose_t_gt[:, -1].reshape(-1)
                gt_pose_local_h[f-(l+1)] = gt_fullpose_local_hmd[:, -1].reshape(-1)
                gt_pose_local_t[f-(l+1)] = gt_fullpose_local_traj[:, -1].reshape(-1)
                gt_contact[f-(l+1)] = gt_contact_window[:, -1].reshape(-1)

                pred_origin[f-(l+1)] = origin_mat_hmd[:, :, :, :3, 1:].reshape(-1)
                pred_fullpose_local_hmd = fromReftoLocal(23, pose_hb, pred_skel)
                pred_pelvis_local_hmd = spineFK2(h_origin_3p_trans[:, :, :9], pred_fullpose_local_hmd)
                pred_fullpose_local_hmd[:, :, 0] = pred_pelvis_local_hmd[:, :, 0]
                pred_fullpose_local_hmd = pred_fullpose_local_hmd[:, :, :, :3, 1:]
                pred_pose[f-(l+1)] = pose_hb[:, -1].reshape(-1)
                pred_pose_local[f-(l+1)] = pred_fullpose_local_hmd[:, -1].reshape(-1)
                pred_contact[f-(l+1)] = contact_hb[:, -1].reshape(-1)
                
                pred_origin_h[f-(l+1)] = origin_mat_hmd[:, :, :, :3, 1:].reshape(-1)
                pred_fullpose_local_hmd_h = fromReftoLocal(23, pose_h, pred_skel)
                pred_pelvis_local_hmd_h = spineFK2(h_origin_3p_trans[:, :, :9], pred_fullpose_local_hmd_h)
                pred_fullpose_local_hmd_h[:, :, 0] = pred_pelvis_local_hmd_h[:, :, 0]
                pred_fullpose_local_hmd_h = pred_fullpose_local_hmd_h[:, :, :, :3, 1:]
                pred_pose_h[f-(l+1)] = pose_h[:, -1].reshape(-1)
                pred_pose_local_h[f-(l+1)] = pred_fullpose_local_hmd_h[:, -1].reshape(-1)
                pred_contact_h[f-(l+1)] = contact_h[:, -1].reshape(-1)
                
                pred_origin_t[f-(l+1)] = origin_mat_traj[:, :, :, :3, 1:].reshape(-1)
                pred_fullpose_local_hmd_t = fromReftoLocal(23, pose_t, pred_skel)
                pred_pelvis_local_hmd_t = spineFK2(t_origin_3p_trans[:, :, :9], pred_fullpose_local_hmd_t)
                pred_fullpose_local_hmd_t[:, :, 0] = pred_pelvis_local_hmd_t[:, :, 0]
                pred_fullpose_local_hmd_t = pred_fullpose_local_hmd_t[:, :, :, :3, 1:]
                pred_pose_t[f-(l+1)] = pose_t[:, -1].reshape(-1)
                pred_pose_local_t[f-(l+1)] = pred_fullpose_local_hmd_t[:, -1].reshape(-1)
                pred_contact_t[f-(l+1)] = contact_t[:, -1].reshape(-1)

            names.append(sample['name'])
            save_csv(gt_inputs.squeeze(), OUTPATH + "/" + sample['name'] + "_input.csv")
            gt_local = torch.cat((gt_origin, gt_pose_local_h, gt_contact), dim=-1)
            save_csv(gt_local.squeeze(), OUTPATH + "/" + sample['name'] + "_gt.csv")

            avg_pelvis_pos_err, avg_pelvis_rot_err, avg_pelv_linvel_err, avg_pelv_angvel_err, \
            avg_joint_pos_err, avg_joint_rot_err, avg_joint_linvel_err, avg_joint_angvel_err, \
            per_joint_pos_err, per_joint_rot_err, per_joint_linvel_err, per_joint_angvel_err, \
            len = inference_err(pred_pose_local, gt_pose_local_h)
            cont_acc = contact_accuracy(pred_contact.unsqueeze(0), gt_contact.unsqueeze(0)).unsqueeze(0)
            ucont_acc = contact_accuracy(pred_contact_h.unsqueeze(0), gt_contact.unsqueeze(0)).unsqueeze(0)
            anim_err = np.array([avg_pelvis_pos_err.item(), avg_pelvis_rot_err.item(), avg_pelv_linvel_err.item(), avg_pelv_angvel_err.item(), 
                                 avg_joint_pos_err.item(), avg_joint_rot_err.item(), avg_joint_linvel_err.item(), avg_joint_angvel_err.item(), cont_acc.item(), len])
            print("p_p: {} | p_q: {} | p_v: {} | p_w: {}".format(anim_err[0], anim_err[1], anim_err[2], anim_err[3]))
            print("j_p: {} | j_q: {} | j_v: {} | j_w: {}".format(anim_err[4], anim_err[5], anim_err[6], anim_err[7]))
            print("cont: {}".format(anim_err[8]))
            total_frames += len

            errs.append(anim_err)
            sum_anim_err = anim_err * len
            sum_anim_err[-1] = len
            sum_errs.append(sum_anim_err)
            per_joint_pos_err = np.append(len, per_joint_pos_err.numpy())
            per_joint_rot_err = np.append(len, per_joint_rot_err.numpy())
            per_joint_linvel_err = np.append(len, per_joint_linvel_err.numpy())
            per_joint_angvel_err = np.append(len, per_joint_angvel_err.numpy())
            pj_pos_errs.append(per_joint_pos_err)
            pj_rot_errs.append(per_joint_rot_err)
            pj_linvel_errs.append(per_joint_linvel_err)                   
            pj_angvel_errs.append(per_joint_angvel_err)                   
            pred_local = torch.cat((pred_origin, pred_pose_local, pred_contact), dim=-1)
            save_csv(pred_local.squeeze(), OUTPATH + "/" + sample['name'] + "_output.csv")

            avg_pelvis_pos_err_h, avg_pelvis_rot_err_h, avg_pelv_linvel_err_h, avg_pelv_angvel_err_h, \
            avg_joint_pos_err_h, avg_joint_rot_err_h, avg_joint_linvel_err_h, avg_joint_angvel_err_h, \
            per_joint_pos_err_h, per_joint_rot_err_h, per_joint_linvel_err_h, per_joint_angvel_err_h, \
            len = inference_err(pred_pose_local_h, gt_pose_local_h)
            cont_acc_h = contact_accuracy(pred_contact_h.unsqueeze(0), gt_contact.unsqueeze(0)).unsqueeze(0)
            anim_err_h = np.array([avg_pelvis_pos_err_h.item(), avg_pelvis_rot_err_h.item(), avg_pelv_linvel_err_h.item(), avg_pelv_angvel_err_h.item(),
                                      avg_joint_pos_err_h.item(), avg_joint_rot_err_h.item(), avg_joint_linvel_err_h.item(), avg_joint_angvel_err_h.item(), cont_acc_h.item(), len])

            errs_h.append(anim_err_h)
            sum_anim_err_h = anim_err_h * len
            sum_anim_err_h[-1] = len
            sum_errs_h.append(sum_anim_err_h)
            per_joint_pos_err_h = np.append(len, per_joint_pos_err_h.numpy())
            per_joint_rot_err_h = np.append(len, per_joint_rot_err_h.numpy())
            per_joint_linvel_err_h = np.append(len, per_joint_linvel_err_h.numpy())
            per_joint_angvel_err_h = np.append(len, per_joint_angvel_err_h.numpy())
            pj_pos_errs_h.append(per_joint_pos_err_h)
            pj_rot_errs_h.append(per_joint_rot_err_h)
            pj_linvel_errs_h.append(per_joint_linvel_err_h)                   
            pj_angvel_errs_h.append(per_joint_angvel_err_h)
            pred_local_h = torch.cat((pred_origin_h, pred_pose_local_h, pred_contact_h), dim=-1)
            save_csv(pred_local_h.squeeze(), OUTPATH + "/" + sample['name'] + "_output_h.csv")

            avg_pelvis_pos_err_t, avg_pelvis_rot_err_t, avg_pelv_linvel_err_t, avg_pelv_angvel_err_t, \
            avg_joint_pos_err_t, avg_joint_rot_err_t, avg_joint_linvel_err_t, avg_joint_angvel_err_t, \
            per_joint_pos_err_t, per_joint_rot_err_t, per_joint_linvel_err_t, per_joint_angvel_err_t, \
            len = inference_err(pred_pose_local_t, gt_pose_local_t)
            cont_acc_t = contact_accuracy(pred_contact_t.unsqueeze(0), gt_contact.unsqueeze(0)).unsqueeze(0)

            anim_err_t = np.array([avg_pelvis_pos_err_t.item(), avg_pelvis_rot_err_t.item(), avg_pelv_linvel_err_t.item(), avg_pelv_angvel_err_t.item(),
                                        avg_joint_pos_err_t.item(), avg_joint_rot_err_t.item(), avg_joint_linvel_err_t.item(), avg_joint_angvel_err_t.item(), cont_acc_t.item(), len])
            
            errs_t.append(anim_err_t)
            sum_anim_err_t = anim_err_t * len
            sum_anim_err_t[-1] = len
            sum_errs_t.append(sum_anim_err_t)
            per_joint_pos_err_t = np.append(len, per_joint_pos_err_t.numpy())
            per_joint_rot_err_t = np.append(len, per_joint_rot_err_t.numpy())
            per_joint_linvel_err_t= np.append(len, per_joint_linvel_err_t.numpy())
            per_joint_angvel_err_t = np.append(len, per_joint_angvel_err_t.numpy())
            pj_pos_errs_t.append(per_joint_pos_err_t)
            pj_rot_errs_t.append(per_joint_rot_err_t)
            pj_linvel_errs_t.append(per_joint_linvel_err_t)                   
            pj_angvel_errs_t.append(per_joint_angvel_err_t)
            pred_local_t = torch.cat((pred_origin_t, pred_pose_local_t, pred_contact_t), dim=-1)
            save_csv(pred_local_t.squeeze(), OUTPATH + "/" + sample['name'] + "_output_t.csv")
        
        total_err = sum(sum_errs) / total_frames
        print("-----inference done-----")
        print("full set errrors!")
        print("p_p: {} | p_q: {} | p_v: {} | p_w: {}".format(total_err[0], total_err[1], total_err[2], total_err[3]))
        print("j_p: {} | j_q: {} | j_v: {} | j_w: {}".format(total_err[4], total_err[5], total_err[6], total_err[7]))
        print("cont: {}".format(total_err[8]))

        err_labels = ['pelvis_pos', 'pelvis_rot', 'pelvis_linvel', 'pelvis_angvel', 'joint_pos', 'joint_rot', 'joint_linvel', 'joint_angvel', 'contact', 'length']
        errs_pd = pd.DataFrame(errs)
        sum_errs_pd = pd.DataFrame(sum_errs)
        errs_pd.index = names
        errs_pd.columns = err_labels
        sum_errs_pd.index = names
        sum_errs_pd.columns = err_labels
        errs_pd.to_csv(OUTPATH + '/errs.csv')
        sum_errs_pd.to_csv(OUTPATH + '/sum_errs.csv')

        errs_pd_h = pd.DataFrame(errs_h)
        sum_errs_pd_h = pd.DataFrame(sum_errs_h)
        errs_pd_h.index = names
        errs_pd_h.columns = err_labels
        sum_errs_pd_h.index = names
        sum_errs_pd_h.columns = err_labels
        errs_pd_h.to_csv(OUTPATH + '/errs_h.csv')
        sum_errs_pd_h.to_csv(OUTPATH + '/sum_errs_h.csv')

        errs_pd_t = pd.DataFrame(errs_t)
        sum_errs_pd_t = pd.DataFrame(sum_errs_t)
        errs_pd_t.index = names
        errs_pd_t.columns = err_labels
        sum_errs_pd_t.index = names
        sum_errs_pd_t.columns = err_labels
        errs_pd_t.to_csv(OUTPATH + '/errs_t.csv')
        sum_errs_pd_t.to_csv(OUTPATH + '/sum_errs_t.csv')

        pj_labels = ["Length", "Hips", "Chest", "Chest2", "Chest3", "Chest4", "Neck", "Head",
          "RightCollar", "RightShoulder", "RightElbow", "RightWrist",
          "LeftCollar", "LeftShoulder", "LeftElbow", "LeftWrist",
          "RightHip", "RightKnee", "RightAnkle", "RightToe",
          "LeftHip", "LeftKnee", "LeftAnkle", "LeftToe"]
        pos_errs_pd = pd.DataFrame(pj_pos_errs)
        rot_errs_pd = pd.DataFrame(pj_rot_errs)
        linvel_errs_pd = pd.DataFrame(pj_linvel_errs)
        angvel_errs_pd = pd.DataFrame(pj_angvel_errs)
        
        pos_errs_pd.index = names
        pos_errs_pd.columns = pj_labels
        rot_errs_pd.index = names
        rot_errs_pd.columns = pj_labels
        linvel_errs_pd.index = names
        linvel_errs_pd.columns = pj_labels
        angvel_errs_pd.index = names
        angvel_errs_pd.columns = pj_labels

        pos_errs_pd.to_csv(OUTPATH + '/pj_pos_errs.csv')
        rot_errs_pd.to_csv(OUTPATH + '/pj_rot_errs.csv')
        linvel_errs_pd.to_csv(OUTPATH + '/pj_linvel_errs.csv')
        angvel_errs_pd.to_csv(OUTPATH + '/pj_angvel_errs.csv')

        pos_errs_pd_h = pd.DataFrame(pj_pos_errs_h)
        rot_errs_pd_h = pd.DataFrame(pj_rot_errs_h)
        linvel_errs_pd_h = pd.DataFrame(pj_linvel_errs_h)
        angvel_errs_pd_h = pd.DataFrame(pj_angvel_errs_h)

        pos_errs_pd_h.index = names
        pos_errs_pd_h.columns = pj_labels
        rot_errs_pd_h.index = names
        rot_errs_pd_h.columns = pj_labels
        linvel_errs_pd_h.index = names
        linvel_errs_pd_h.columns = pj_labels
        angvel_errs_pd_h.index = names
        angvel_errs_pd_h.columns = pj_labels
        
        pos_errs_pd_h.to_csv(OUTPATH + '/pj_pos_errs_h.csv')
        rot_errs_pd_h.to_csv(OUTPATH + '/pj_rot_errs_h.csv')
        linvel_errs_pd_h.to_csv(OUTPATH + '/pj_linvel_errs_h.csv')
        angvel_errs_pd_h.to_csv(OUTPATH + '/pj_angvel_errs_h.csv')

        pos_errs_pd_t = pd.DataFrame(pj_pos_errs_t)
        rot_errs_pd_t = pd.DataFrame(pj_rot_errs_t)
        linvel_errs_pd_t = pd.DataFrame(pj_linvel_errs_t)
        angvel_errs_pd_t = pd.DataFrame(pj_angvel_errs_t)
        
        pos_errs_pd_t.index = names
        pos_errs_pd_t.columns = pj_labels
        rot_errs_pd_t.index = names
        rot_errs_pd_t.columns = pj_labels
        linvel_errs_pd_t.index = names
        linvel_errs_pd_t.columns = pj_labels
        angvel_errs_pd_t.index = names
        angvel_errs_pd_t.columns = pj_labels

        pos_errs_pd_t.to_csv(OUTPATH + '/pj_pos_errs_t.csv')
        rot_errs_pd_t.to_csv(OUTPATH + '/pj_rot_errs_t.csv')
        linvel_errs_pd_t.to_csv(OUTPATH + '/pj_linvel_errs_t.csv')
        angvel_errs_pd_t.to_csv(OUTPATH + '/pj_angvel_errs_t.csv')
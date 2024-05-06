import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

red_parents = [-1, 0, 1, 2, 3, 4,
                4, 6, 7,
                4, 9, 10,
                0, 12, 13,
                0, 15, 16]

full_parents = [-1, 0, 1, 2, 3, 4, 5,
                4, 7, 8, 9,
                4, 11, 12, 13,
                0, 15, 16, 17,
                0, 19, 20, 21]

full_parents_smpl = [-1, 0, 1, 2, 3, 4,
                3, 6, 7, 8,
                3, 10, 11, 12,
                0, 14, 15, 16,
                0, 18, 19, 20]

rleg_idx = [14, 15, 16, 17]
lleg_idx = [18, 19, 20, 21]

spine_idx = [5, 4, 3, 2, 1, 0]
spine_idx2 = [6, 5, 4, 3, 2, 1]
spine_idx_smpl = [5, 4, 3, 2, 1]

@torch.no_grad()
def spineFK(head_trans, pose, pelvis_rot):
    pose = to4x4from9d(22, pose)
    spine = pose[:, 0, spine_idx, :, :]
    spine_inv = torch.inverse(spine)
    pelvis_trans = head_trans
    for i in range(6):
        pelvis_trans = pelvis_trans @ spine_inv[:, i]
    pelvis_trans = pelvis_trans[:, :3, 1:]
    pelvis_trans[:, :, :2] = pelvis_rot.reshape(-1, 3, 2)
    return pelvis_trans.reshape((-1, 1, 9))

@torch.no_grad()
def spineFK2(head_trans, pose):
    b, l, _, _, _ = pose.shape
    spine = pose[:, :, spine_idx2]
    spine_inv = torch.inverse(spine)
    pelvis_trans = to4x4from9d(1, head_trans)
    for i in range(6):
        pelvis_trans = pelvis_trans @ spine_inv[:, :, i].unsqueeze(2)
    pelvis_trans[:, :, :, :3, :3] = pose[:, :, 0, :3, :3].unsqueeze(2)
    return pelvis_trans

@torch.no_grad()
def spineFKSMPL(head_trans, pose):
    b, l, _, _, _ = pose.shape
    spine = pose[:, :, spine_idx_smpl]
    spine_inv = torch.inverse(spine)
    pelvis_trans = to4x4from9d(1, head_trans)
    for i in range(5):
        pelvis_trans = pelvis_trans @ spine_inv[:, :, i].unsqueeze(2)
    pelvis_trans[:, :, :, :3, :3] = pose[:, :, 0, :3, :3].unsqueeze(2)
    return pelvis_trans

@torch.no_grad()
def spineFK3(head_trans, pose):
    b, l, c = pose.shape
    j = c//6
    pose = to3x3from6d(j, pose)
    spine = pose[:, :, spine_idx2]
    spine_inv = torch.inverse(spine)
    pelvis_trans = to4x4from9d(1, head_trans)
    for i in range(6):
        pelvis_trans = pelvis_trans @ spine_inv[:, :, i].unsqueeze(2)
    pelvis_trans[:, :, :, :3, :3] = pose[:, :, 0, :3, :3].unsqueeze(2)
    return pelvis_trans

def toeFK(ref_trans, pose):
    b, l, c  = pose.shape
    j = c//9
    pose = to4x4from9d(j, pose).squeeze(1)
    rtoe = ref_trans.clone()
    ltoe = ref_trans.clone()
    for i in rleg_idx:
        rtoe = rtoe @ pose[:, i]
    for i in lleg_idx:
        ltoe = ltoe @ pose[:, i]
    rtoe = rtoe[:, :3, 3].reshape(b, l, 3)
    ltoe = ltoe[:, :3, 3].reshape(b, l, 3)
    toe = torch.cat((rtoe, ltoe), dim=-1)
    return toe

def FK22(output):
    pose = to4x4from9d(22, output)
    g_pose = pose.clone()
    for j in range(1, 22):
        parent = g_pose[:, :, full_parents_smpl[j]].clone()
        g_pose[:, :, j] = parent @ pose[:, :, j]
    return g_pose[:, :, :, :3, 1:]

def FK23(output):
    pose = to4x4from9d(23, output)
    g_pose = pose.clone()
    for j in range(1, 23):
        parent = g_pose[:, :, full_parents[j]].clone()
        g_pose[:, :, j] = parent @ pose[:, :, j]
    return g_pose[:, :, :, :3, 1:]

def to3x3from6d(j, basis):
    if basis.dim() == 2:
        basis = basis.unsqueeze(0)
    b, l, _ = basis.shape
    mat = basis.reshape(b, l, j, 3, 2)
    x = torch.cross(mat[:, :, :, :, 0], mat[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    mat = torch.cat((x, mat), dim=-1)
    return mat

def to4x4from6d(j, basis, offset):
    if basis.dim() == 2:
        basis = basis.unsqueeze(0)
    b, l, _ = basis.shape
    mat = basis.reshape(b, l, j, 3, 2)
    offset = offset.reshape(b, l, j, 3, 1)
    mat = torch.cat([mat, offset], dim=-1)
    x = torch.cross(mat[:, :, :, :, 0], mat[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    mat = torch.cat((x, mat), dim=-1)    
    mat = torch.cat((mat, torch.tensor([0, 0, 0, 1]).repeat(b, l, j, 1, 1).to(mat.device)), dim=3)
    return mat

def to4x4from9d(j, basis):
    if basis.dim() == 2:
        basis = basis.unsqueeze(0)
    b, l, _ = basis.shape
    mat = basis.reshape(b, l, j, 3, 3)
    x = torch.cross(mat[:, :, :, :, 0], mat[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    mat = torch.cat((x, mat), dim=-1)    
    mat = torch.cat((mat, torch.tensor([0, 0, 0, 1]).repeat(b, l, j, 1, 1).to(mat.device)), dim=3)
    return mat

def fromReftoLocal_18(j, ref, offset):
    if ref.dim() == 2:
        ref = ref.unsqueeze(0)
    b, l, _ = ref.shape
    mat = ref.reshape(b, l, j, 3, 3)
    x = torch.cross(mat[:, :, :, :, 0], mat[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    mat = torch.cat((x, mat), dim=-1)    
    local = mat.clone()
    for i in range(j):
        parent = red_parents[i]
        if parent != -1:
            local[:, :, i, :3, :3] = torch.matmul(torch.inverse(mat[:, :, parent, :3, :3]), mat[:, :, i, :3, :3])
    offset = offset.reshape(b, l, j, 3, 1)[:, :, 1:].squeeze(-1)
    local[:, :, 1:, :3, 3] = offset
    local = torch.cat((local, torch.tensor([0, 0, 0, 1]).repeat(b, l, j, 1, 1).to(local.device)), dim=3)
    return local

def fromReftoLocal(j, ref, offset):
    if ref.dim() == 2:
        ref = ref.unsqueeze(0)
    b, l, _ = ref.shape
    mat = ref.reshape(b, l, j, 3, 3)
    x = torch.cross(mat[:, :, :, :, 0], mat[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    mat = torch.cat((x, mat), dim=-1)    
    local = mat.clone()
    for i in range(j):
        parent = full_parents[i]
        if parent != -1:
            local[:, :, i, :3, :3] = torch.matmul(torch.inverse(mat[:, :, parent, :3, :3]), mat[:, :, i, :3, :3])
    offset = offset.reshape(b, l, j, 3, 1)[:, :, 1:].squeeze(-1)
    local[:, :, 1:, :3, 3] = offset
    local = torch.cat((local, torch.tensor([0, 0, 0, 1]).repeat(b, l, j, 1, 1).to(local.device)), dim=3)
    return local

def fromReftoLocal2(j, full, offset):
    if full.dim() == 2:
        full = full.unsqueeze(0)
    b, l, _ = full.shape
    mat = full.reshape(b, l, j, 3, 2)
    x = torch.cross(mat[:, :, :, :, 0], mat[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    mat = torch.cat((x, mat), dim=-1)    
    local = mat.clone()
    for i in range(j):
        parent = full_parents[i]
        if parent != -1:
            local[:, :, i] = torch.matmul(torch.inverse(mat[:, :, parent]), mat[:, :, i])
    offset = offset.reshape(b, l, j, 3, 1)
    local = torch.cat([local, offset], dim=-1)
    local = torch.cat((local, torch.tensor([0, 0, 0, 1]).repeat(b, l, j, 1, 1).to(local.device)), dim=3)
    return local

def fromReftoLocalSMPL(j, ref, offset):
    if ref.dim() == 2:
        ref = ref.unsqueeze(0)
    b, l, _ = ref.shape
    mat = ref.reshape(b, l, j, 3, 3)
    x = torch.cross(mat[:, :, :, :, 0], mat[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    mat = torch.cat((x, mat), dim=-1)    
    local = mat.clone()
    for i in range(j):
        parent = full_parents_smpl[i]
        if parent != -1:
            local[:, :, i, :3, :3] = torch.matmul(torch.inverse(mat[:, :, parent, :3, :3]), mat[:, :, i, :3, :3])
    offset = offset.reshape(b, l, j, 3, 1)[:, :, 1:].squeeze(-1)
    local[:, :, 1:, :3, 3] = offset
    local = torch.cat((local, torch.tensor([0, 0, 0, 1]).repeat(b, l, j, 1, 1).to(local.device)), dim=3)
    return local

def toLocalFull(j, full, offset):
    if full.dim() == 2:
        full = full.unsqueeze(0)
    b, l, _ = full.shape
    mat = full.reshape(b, l, j, 3, 2)
    x = torch.cross(mat[:, :, :, :, 0], mat[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    mat = torch.cat((x, mat), dim=-1)    
    local = mat.clone()
    offset = offset.reshape(b, l, j, 3, 1)
    local = torch.cat([local, offset], dim=-1)
    local = torch.cat((local, torch.tensor([0, 0, 0, 1]).repeat(b, l, j, 1, 1).to(local.device)), dim=3)
    return local

def changeBasis(num_joints, s_input, s_basis, t_basis):
    # t_basis_inv @ s_basis @ input
    b, l, _ = s_input.shape
    s_basis_mat = s_basis.repeat(1, l, num_joints, 1, 1)
    t_basis_inv_mat = torch.inverse(t_basis).repeat(1, l, num_joints, 1, 1)
    b_input = (t_basis_inv_mat @ (s_basis_mat @ to4x4from9d(num_joints, s_input)))[:, :, :, :3, 1:].reshape(b, l, -1)
    return b_input

def kl_div(mu, logvar):
    loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return loss

def angvel_loss(output, target):
    b, l, c = output.shape
    j = c // 6
    output3x3 = to3x3from6d(j, output)
    target3x3 = to3x3from6d(j, target)
    avel = torch.inverse(output3x3[:, :-1]) @ output3x3[:, 1:]
    avel_target = torch.inverse(target3x3[:, :-1]) @ target3x3[:, 1:]
    loss = torch.mean(torch.cosine_similarity(avel[:, :, :, 1:], avel_target[:, :, :, 1:], dim = -2))
    return loss

def cont_consistency(j, pred_pose, pred_contact, offset):
    b, l, _ = pred_pose.shape
    pred_pose_local = fromReftoLocal(j, pred_pose, offset)[:, :, :, :3, 1:].reshape(b, l, -1)
    pred_pose_fk = FK23(pred_pose_local)
    positions = pred_pose_fk.reshape(b, l, j, 3, 3)[:, :, :, :, 2]
    velocities = positions[:, 1:] - positions[:, :-1]
    velocities = velocities[:, :, [j-5, j-1]]
    contacts = pred_contact[:, 1:].reshape(b, l-1, 2, 1)
    loss = torch.mean(torch.linalg.norm((velocities * contacts), dim=-1))
    return loss

def cont_consistency2(j, pred_pose, pred_contact, offset, gt_pose):
    b, l, _ = pred_pose.shape
    pred_pose_local = fromReftoLocal(j, pred_pose, offset)[:, :, :, :3, 1:].reshape(b, l, -1)
    pred_pose_fk = FK23(pred_pose_local)
    gt_pose_local = fromReftoLocal(j, gt_pose, offset)[:, :, :, :3, 1:].reshape(b, l, -1)
    gt_pose_fk = FK23(gt_pose_local)
    positions = pred_pose_fk.reshape(b, l, j, 3, 3)[:, :, :, :, 2]
    gt_positions = gt_pose_fk.reshape(b, l, j, 3, 3)[:, :, :, :, 2]
    diff = positions - gt_positions
    fk_loss = torch.mean(torch.linalg.norm(diff, dim=-1))
    velocities = positions[:, 1:] - positions[:, :-1]
    velocities = velocities[:, :, [j-5, j-1]]
    contacts = pred_contact[:, 1:].reshape(b, l-1, 2, 1)
    contconsit_loss = torch.mean(torch.linalg.norm((velocities * contacts), dim=-1))
    return fk_loss, contconsit_loss

@torch.no_grad()
def pelv_err(output, target):
    b, l, _ = output.shape
    output = output.reshape(b,l,3,3)
    target = target.reshape(b,l,3,3)
    pos = output[:, :, :3, 2]
    gt_pos = target[:, :, :3, 2]
    rot = output[:, :, :3, :2]
    gt_rot = target[:, :, :3, :2]
    
    pos_err = torch.linalg.norm(pos - gt_pos, dim=-1)
    rot_err = torch.rad2deg(torch.arccos(torch.clamp(torch.cosine_similarity(rot, gt_rot, dim = -2), -1.0, 1.0)))
    avg_pos_err = torch.mean(pos_err)
    avg_rot_err = torch.mean(rot_err)
    return avg_pos_err, avg_rot_err

@torch.no_grad()
def pelv_err2(output, target):
    b, l, c = output.shape
    output = output.reshape(b,l,3)
    target = target.reshape(b,l,3)
    
    pos_err = torch.linalg.norm(output - target, dim=-1)    
    avg_pos_err = torch.mean(pos_err)
    return avg_pos_err, pos_err

@torch.no_grad()
def joint_err(output, target):
    b, l, c = output.shape
    j = c//9
    output = output.reshape(b,l,j,3,3)
    target = target.reshape(b,l,j,3,3)
    pos = output[:, :, :, :3, 2]
    gt_pos = target[:, :, :, :3, 2]
    rot = output[:, :, :, :3, :2]
    gt_rot = target[:, :, :, :3, :2]

    pos_err = torch.linalg.norm(pos - gt_pos, dim=-1)
    rot_err = torch.rad2deg(torch.arccos(torch.clamp(torch.cosine_similarity(rot, gt_rot, dim = -2), -1.0, 1.0)))
    per_joint_pos_err = torch.mean(pos_err, dim=(0, 1))
    per_joint_rot_err = torch.mean(rot_err, dim=(0, 1, 3))
    avg_joint_pos_err = torch.mean(pos_err)
    avg_joint_rot_err = torch.mean(rot_err)

    return avg_joint_pos_err, avg_joint_rot_err, per_joint_pos_err, per_joint_rot_err

@torch.no_grad()
def joint_err2(output, target):
    b, l, c = output.shape
    j = c//6
    rot = output.reshape(b,l,j,3,2)
    gt_rot = target.reshape(b,l,j,3,2)
    rot_err = torch.rad2deg(torch.arccos(torch.clamp(torch.cosine_similarity(rot, gt_rot, dim = -2), -1.0, 1.0)))
    per_joint_rot_err = torch.mean(rot_err, dim=(0,1,3))
    avg_joint_rot_err = torch.mean(rot_err)
    return avg_joint_rot_err, per_joint_rot_err

def l2_3d(output, target):
    if output.dim() == 2:
        output = output.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    b, l, c = output.shape
    j = c//3
    output = output.reshape(b, l, j, 3)
    target = target.reshape(b, l, j, 3)
    dist = torch.norm(output - target, dim = -1)
    avg_dist = torch.mean(dist)
    return avg_dist, dist

def l2_3d_vel(output, target):
    if output.dim() == 2:
        output = output.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    b, l, c = output.shape
    j = c//3
    output = output.reshape(b, l, j, 3)
    target = target.reshape(b, l, j, 3)
    output = output[:, 1:] - output[:, :-1]
    target = target[:, 1:] - target[:, :-1]
    l = l - 1
    dist = torch.norm(output - target, dim = -1)
    avg_dist = torch.mean(dist)
    return avg_dist, dist

@torch.no_grad()
def trans_err(output, target):
    if output.dim() == 2:
        output = output.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    b, l, c = output.shape
    j = c//9
    trans = output.reshape(b, l, j, 3, 3)
    trans_gt = target.reshape(b, l, j, 3, 3)
    pos = trans[:, :, :, :3, -1]
    gt_pos = trans_gt[:, :, :, :3, -1]
    rot = trans[:, :, :, :3, :-1]
    gt_rot = trans_gt[:, :, :, :3, :-1]

    pos_err = torch.linalg.norm(pos - gt_pos, dim=-1)
    rot_err = torch.rad2deg(torch.arccos(torch.clamp(torch.cosine_similarity(rot, gt_rot, dim = -2), -1.0, 1.0)))
    per_joint_pos_err = torch.mean(pos_err, dim=(0, 1))
    per_joint_rot_err = torch.mean(rot_err, dim=(0, 1, 3))
    avg_joint_pos_err = torch.mean(pos_err)
    avg_joint_rot_err = torch.mean(rot_err)

    return avg_joint_pos_err, avg_joint_rot_err, per_joint_pos_err, per_joint_rot_err, l

@torch.no_grad()
def trans_vel_err(output, target, ref):
    if output.dim() == 2:
        output = output.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    b, l, c = output.shape
    j = c//9
    ref_inv = torch.inverse(ref)
    ref_inv = ref_inv.repeat(1, l-1, j, 1, 1)
    ref = ref.repeat(1, l, j, 1, 1)
    trans = ref @ to4x4from9d(j, output)
    gt_vel = to4x4from9d(j, target)
    vel = (trans[:, 1:] @ torch.inverse(trans[:, :-1]))
    vel = ref_inv @ vel
    gt_vel = ref_inv @ gt_vel
    return F.mse_loss(vel[:, :, :, :3, 1:].reshape(b, l-1, -1), gt_vel[:, :, :, :3, 1:].reshape(b, l-1, -1))

@torch.no_grad()
def contact_accuracy(output, target):
    out_contact = (output>0.5).float()
    accuracy = 1.0 - (torch.count_nonzero(out_contact - target) / torch.numel(output))
    return accuracy

@torch.no_grad()
def inference_err(output, target):
    if output.dim() == 2:
        output = output.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    b, l, c = output.shape
    j = c//9

    g_output = FK23(output)
    g_target = FK23(target)

    output = output.reshape(b,l,j,3,3)
    target = target.reshape(b,l,j,3,3)

    rot = output[:, :, :, :3, :2]
    gt_rot = target[:, :, :, :3, :2]

    g_pos = g_output[:, :, :, :3, 2]
    g_gt_pos = g_target[:, :, :, :3, 2]

    pos_err = torch.linalg.norm(g_pos - g_gt_pos, dim=-1)
    rot_err = torch.rad2deg(torch.arccos(torch.clamp(torch.cosine_similarity(rot, gt_rot, dim = -2), -1.0, 1.0)))

    per_joint_pos_err = torch.mean(pos_err, dim=(0, 1))
    per_joint_rot_err = torch.mean(rot_err, dim=(0, 1, 3))

    avg_pelvis_pos_err = torch.mean(pos_err[:, :, 0])
    avg_pelvis_rot_err = torch.mean(rot_err[:, :, 0])

    avg_joint_pos_err = torch.mean(pos_err[:, :, 1:])
    avg_joint_rot_err = torch.mean(rot_err[:, :, 1:])

    # joint linear velocity
    linvel = g_pos[:, 1:] - g_pos[:, :-1]
    gt_linvel = g_gt_pos[:, 1:] - g_gt_pos[:, :-1]
    linvel_err = torch.linalg.norm(linvel - gt_linvel, dim=-1)
    per_joint_linvel_err = torch.mean(linvel_err, dim=(0, 1))
    avg_pelv_linvel_err = torch.mean(linvel_err[:, :, 0])
    avg_joint_linvel_err = torch.mean(linvel_err[:, :, 1:])

    # joint angular velocity
    x = torch.cross(rot[:, :, :, :, 0], rot[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    rot_mat = torch.cat((x, rot), dim=-1)    
    gt_x = torch.cross(gt_rot[:, :, :, :, 0], gt_rot[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    gt_rot_mat = torch.cat((gt_x, gt_rot), dim=-1)

    angvel = torch.inverse(rot_mat[:, :-1]) @ rot_mat[:, 1:]
    gt_angvel = torch.inverse(gt_rot_mat[:, :-1]) @ gt_rot_mat[:, 1:]

    angvel_err = torch.rad2deg(torch.arccos(torch.clamp(torch.cosine_similarity(angvel[:, :, :, :, 1:], gt_angvel[:, :, :, :, 1:], dim = -2), -1.0, 1.0)))
    per_joint_angvel_err = torch.mean(angvel_err, dim=(0, 1, 3))
    avg_pelv_angvel_err = torch.mean(angvel_err[:, :, 0])
    avg_joint_angvel_err = torch.mean(angvel_err[:, :, 1:])

    return avg_pelvis_pos_err, avg_pelvis_rot_err, avg_pelv_linvel_err, avg_pelv_angvel_err, \
           avg_joint_pos_err, avg_joint_rot_err, avg_joint_linvel_err, avg_joint_angvel_err, \
           per_joint_pos_err, per_joint_rot_err, per_joint_linvel_err, per_joint_angvel_err, l

@torch.no_grad()
def inference_err_SMPL(output, target):
    if output.dim() == 2:
        output = output.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    b, l, c = output.shape
    j = c//9

    g_output = FK22(output)
    g_target = FK22(target)

    output = output.reshape(b,l,j,3,3)
    target = target.reshape(b,l,j,3,3)

    rot = output[:, :, :, :3, :2]
    gt_rot = target[:, :, :, :3, :2]

    g_pos = g_output[:, :, :, :3, 2]
    g_gt_pos = g_target[:, :, :, :3, 2]

    pos_err = torch.linalg.norm(g_pos - g_gt_pos, dim=-1)
    rot_err = torch.rad2deg(torch.arccos(torch.clamp(torch.cosine_similarity(rot, gt_rot, dim = -2), -1.0, 1.0)))

    per_joint_pos_err = torch.mean(pos_err, dim=(0, 1))
    per_joint_rot_err = torch.mean(rot_err, dim=(0, 1, 3))

    avg_pelvis_pos_err = torch.mean(pos_err[:, :, 0])
    avg_pelvis_rot_err = torch.mean(rot_err[:, :, 0])

    avg_joint_pos_err = torch.mean(pos_err[:, :, 1:])
    avg_joint_rot_err = torch.mean(rot_err[:, :, 1:])

    # joint linear velocity
    linvel = g_pos[:, 1:] - g_pos[:, :-1]
    gt_linvel = g_gt_pos[:, 1:] - g_gt_pos[:, :-1]
    linvel_err = torch.linalg.norm(linvel - gt_linvel, dim=-1)
    per_joint_linvel_err = torch.mean(linvel_err, dim=(0, 1))
    avg_pelv_linvel_err = torch.mean(linvel_err[:, :, 0])
    avg_joint_linvel_err = torch.mean(linvel_err[:, :, 1:])

    # joint angular velocity
    x = torch.cross(rot[:, :, :, :, 0], rot[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    rot_mat = torch.cat((x, rot), dim=-1)    
    gt_x = torch.cross(gt_rot[:, :, :, :, 0], gt_rot[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    gt_rot_mat = torch.cat((gt_x, gt_rot), dim=-1)

    angvel = torch.inverse(rot_mat[:, :-1]) @ rot_mat[:, 1:]
    gt_angvel = torch.inverse(gt_rot_mat[:, :-1]) @ gt_rot_mat[:, 1:]

    angvel_err = torch.rad2deg(torch.arccos(torch.clamp(torch.cosine_similarity(angvel[:, :, :, :, 1:], gt_angvel[:, :, :, :, 1:], dim = -2), -1.0, 1.0)))
    per_joint_angvel_err = torch.mean(angvel_err, dim=(0, 1, 3))
    avg_pelv_angvel_err = torch.mean(angvel_err[:, :, 0])
    avg_joint_angvel_err = torch.mean(angvel_err[:, :, 1:])

    return avg_pelvis_pos_err, avg_pelvis_rot_err, avg_pelv_linvel_err, avg_pelv_angvel_err, \
           avg_joint_pos_err, avg_joint_rot_err, avg_joint_linvel_err, avg_joint_angvel_err, \
           per_joint_pos_err, per_joint_rot_err, per_joint_linvel_err, per_joint_angvel_err, l

@torch.no_grad()
def inference_err2(output, target):
    if output.dim() == 2:
        output = output.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    b, l, c = output.shape
    j = c//9

    g_output = FK23(output)
    g_target = FK23(target)

    output = output.reshape(b,l,j,3,3)
    target = target.reshape(b,l,j,3,3)
    rot = output[:, :, :, :3, :2]
    gt_rot = target[:, :, :, :3, :2]

    g_pos = g_output[:, :, :, :3, 2]
    g_gt_pos = g_target[:, :, :, :3, 2]

    pos_err = torch.linalg.norm(g_pos - g_gt_pos, dim=-1)
    rot_err = torch.rad2deg(torch.arccos(torch.cosine_similarity(rot, gt_rot, dim = -2)))

    per_joint_pos_err = torch.mean(pos_err, dim=(0, 1))
    per_joint_rot_err = torch.mean(rot_err, dim=(0, 1, 3))

    avg_pelvis_pos_err = torch.mean(pos_err[:, :, 0])
    avg_pelvis_rot_err = torch.mean(rot_err[:, :, 0])

    avg_joint_pos_err = torch.mean(pos_err[:, :, 1:])
    avg_joint_rot_err = torch.mean(rot_err[:, :, 1:])

    # joint linear velocity
    linvel = g_pos[:, 1:] - g_pos[:, :-1]
    gt_linvel = g_gt_pos[:, 1:] - g_gt_pos[:, :-1]
    linvel_err = torch.linalg.norm(linvel - gt_linvel, dim=-1)
    per_joint_linvel_err = torch.mean(linvel_err, dim=(0, 1))
    avg_pelv_linvel_err = torch.mean(linvel_err[:, :, 0])
    avg_joint_linvel_err = torch.mean(linvel_err[:, :, 1:])

    # joint angular velocity
    x = torch.cross(rot[:, :, :, :, 0], rot[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    rot_mat = torch.cat((x, rot), dim=-1)    
    gt_x = torch.cross(gt_rot[:, :, :, :, 0], gt_rot[:, :, :, :, 1], dim=-1).unsqueeze(-1)
    gt_rot_mat = torch.cat((gt_x, gt_rot), dim=-1)

    angvel = torch.inverse(rot_mat[:, :-1]) @ rot_mat[:, 1:]
    gt_angvel = torch.inverse(gt_rot_mat[:, :-1]) @ gt_rot_mat[:, 1:]

    angvel_err = torch.rad2deg(torch.arccos(torch.clamp(torch.cosine_similarity(angvel[:, :, :, :, 1:], gt_angvel[:, :, :, :, 1:], dim = -2), -1.0, 1.0)))
    per_joint_angvel_err = torch.mean(angvel_err, dim=(0, 1, 3))
    avg_pelv_angvel_err = torch.mean(angvel_err[:, :, 0])
    avg_joint_angvel_err = torch.mean(angvel_err[:, :, 1:])

    return avg_pelvis_pos_err, avg_pelvis_rot_err, avg_pelv_linvel_err, avg_pelv_angvel_err, \
           avg_joint_pos_err, avg_joint_rot_err, avg_joint_linvel_err, avg_joint_angvel_err, \
           per_joint_pos_err, per_joint_rot_err, per_joint_linvel_err, per_joint_angvel_err, l

@torch.no_grad()
def save_csv(tensor, path):
    arr = tensor.cpu().detach().numpy()
    np.savetxt(path, arr, fmt='%1.3f', delimiter=",")
    return

def save_arg(path, args):
    with open(path, 'w') as f:
        json.dump(vars(args), f)
    return

def read_json_as_arg(path):
    with open(path, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
    return t_args
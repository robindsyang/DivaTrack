import os
import pickle
import re
import csv
from uuid import NAMESPACE_URL
import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from mvnx import MVNX

head_idx = 6
IMU_a_idx = [6, 7, 8, 15, 16, 17, 27, 28, 29]

def copy(self):
    cls = self.__class__
    result = cls.__new__(cls)
    result.__dict__.update(self.__dict__)
    return result

def deepcopy(self, memo):
    cls = self.__class__
    result = cls.__new__(cls)
    memo[id(self)] = result
    for k, v in self.__dict__.items():
        setattr(result, k, deepcopy(v, memo))
    return result

def RPY2Quat(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qx, qy, qz, qw]

def XZProjection(mat):
    out = np.eye(4)
    basis_z = np.array([mat[0, 2], 0, mat[2, 2]])
    basis_y = np.array([[0, 1, 0]])
    basis_x = np.cross(basis_y, basis_z)
    pos = np.array([mat[0, 3], 0, mat[2, 3]])
    
    out[:3, 0] = basis_x / np.linalg.norm(basis_x)
    out[:3, 1] = basis_y / np.linalg.norm(basis_y)
    out[:3, 2] = basis_z / np.linalg.norm(basis_z)
    out[:3, 3] = pos
    return out

class Animation:
    def __init__(self):
        self.name = None
        self.coord = None
        self.fps = 0
        self.length = 0
        self.joints = []
        self.parents = []
        self.local_t = None
        self.world_t = None
        self.ref_t = None
        self.world_vw = None
        self.ref_vw = None
        self.imu_a = None
        self.contact = None
        self.env = None

    def load_bvh(self, path, euler = 'YXZ', trg_coord='left', shift=0):
        base = os.path.basename(path)
        self.name = os.path.splitext(base)[0]
        bvh = open(path + '.bvh', 'r')
        
        pose = []
        offsets = []
        current_joint = 0
        end_site = False
        for line in bvh:
            joint_line = re.match("ROOT\s+(\w+)", line)
            if joint_line == None:
                joint_line = re.match("\s*JOINT\s+(\w+)", line)

            if joint_line:
                self.joints.append(joint_line.group(1))
                self.parents.append(current_joint)
                current_joint = len(self.parents) - 1
                continue

            endsite_line = re.match("\s*End\sSite", line)
            if endsite_line:
                end_site = True
                continue

            offset_line = re.match("\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offset_line:
                if not end_site:
                    offsets.append(np.array([offset_line.group(1), offset_line.group(2), offset_line.group(3)]))
                continue

            if "}" in line:
                if end_site:
                    end_site = False
                else:
                    current_joint = self.parents[current_joint]
                continue

            if "Frames" in line:
                self.length = int(line.split(' ')[-1])
                continue

            if "Frame Time:" in line:
                self.fps = int(1 / float(line.split(' ')[-1]))
                continue

            if "HIERARCHY" in line or "{" in line or "CHANNELS" in line or "MOTION" in line:
                continue

            # pose.append(np.array(line.strip().split('    ')))
            pose.append(np.array(line.strip().split(' ')))
        
        self.coord = trg_coord
        self.joints = np.asarray(self.joints, dtype=str)
        self.parents = np.asarray(self.parents, dtype=np.int8)
        
        pose = np.asarray(pose, dtype=np.float32)
        offsets = np.asarray(offsets, dtype=np.float32)
        
        # + 1 for root pos ori + shift
        pose = pose.reshape(self.length, self.joints.shape[0] + 1, 3)[shift:]
        self.length = pose.shape[0]

        # to left-handed coordinate system (unity)
        if trg_coord == 'left':
            # x -> -x for positions
            offsets[:, 0] *= -1
            pose[:, 0, 0] *= -1
            # z x y -> -z x -y for rotation (right -> left)
            pose[:, 1:, [0, 2]] *= -1

        self.local_t = np.zeros((self.length, self.joints.shape[0], 4, 4))
        for f in range(self.length):
            for j in range(1, self.joints.shape[0] + 1):
                local_t_mat = np.eye(4)
                r = R.from_euler(euler, pose[f, j], degrees=True)
                p = pose[f, 0] if j == 1 else offsets[j - 1]
                local_t_mat[:3, 3] = p
                local_t_mat[:3, :3] = r.as_matrix()
                self.local_t[f, j - 1] = local_t_mat

    def load_imu_mvnx(self, path):
        self.contact = np.zeros((self.length, 2))
        self.imu_a = np.zeros((self.length, 3, 3))
        
        if not os.path.exists(path + '.mvnx'):
            return
        
        imu = MVNX(path + '.mvnx', 'r')
        for i in range(self.length): # 0 right 1 left
            if imu.footContacts[i, 2] == 1 or imu.footContacts[i, 3] == 1: self.contact[i, 0] = 1
            if imu.footContacts[i, 0] == 1 or imu.footContacts[i, 1] == 1: self.contact[i, 1] = 1
        self.imu_a = imu.sensorFreeAcceleration[:, IMU_a_idx].reshape((self.length, 3, 3))
        for i in range(self.length):
            for j in range(3):
                self.imu_a[i, j] = self.imu_a[i, j, [1, 2, 0]]
                self.imu_a[i, j, 0] *= -1
        
    def load_env(self, path):
        rows = []
        
        if not os.path.exists(path + '.csv'):
            return
        
        with open(path + '.csv', 'r') as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                row.remove(" ")
                rows.append(row)
        nH = int(header[0])
        nR = int(header[1])
        data = np.asarray(rows, dtype=np.float32)
        length = data.shape[0]
        self.env = data.reshape(length, nH, -1)
    
    def upsample2(self):
        length, nJ, _, _ = self.local_t.shape
        interp_local_t_list = []
        interp_imu_list = []
        interp_contact_list = []
        
        new_length = length * 2 - 1
        for f in range(length-1):
            for j in range(nJ):
                interp_local_t_list.append(self.local_t[f, j])

            for j in range(nJ):
                key_times = [0, 1]
                key_rots = R.from_matrix(self.local_t[f:f+2, j, :3, :3])
                slerp = Slerp(key_times, key_rots)
                interp_R = slerp(0.5)
                interp_p = (self.local_t[f, j, :3, 3] + self.local_t[f+1, j, :3, 3]) / 2
                interp_T = np.eye(4)
                interp_T[:3, :3] = interp_R.as_matrix()
                interp_T[:3, 3] = interp_p
                interp_local_t_list.append(interp_T)

            if f==length-2:
                for j in range(nJ):
                    interp_local_t_list.append(self.local_t[f+1,j])

            for j in range(3):
                interp_imu_list.append(self.imu_a[f, j])

            for j in range(3):
                interp_imu_a = (self.imu_a[f, j] + self.imu_a[f+1, j]) / 2
                interp_imu_list.append(interp_imu_a)

            if f==length-2:
                for j in range(3):
                    interp_imu_list.append(self.imu_a[f+1, j])

            for j in range(2):
                interp_contact_list.append(self.contact[f, j])

            for j in range(2):
                interp_contact_list.append(self.contact[f, j])

            if f==length-2:
                for j in range(2):
                    interp_contact_list.append(self.contact[f+1, j])

        self.length = new_length
        self.local_t = np.asarray(interp_local_t_list).reshape((new_length, nJ, 4, 4))
        self.imu_a = np.asarray(interp_imu_list).reshape((new_length, 3, 3))
        self.contact = np.asarray(interp_contact_list).reshape((new_length, 2))

    def downsample(self, ratio):
        self.local_t = self.local_t[::ratio]
        self.imu_a = self.imu_a[:, ::ratio]
        self.contact = self.contact[::ratio]
        self.length = self.local_t.shape[0]

    def delete_joints(self, joint_names):
        for joint in joint_names:
            if joint in self.joints:
                idx = np.where(self.joints == joint)[0][0]
                parent_idx = self.parents[idx]
                children_indices = np.where(self.parents == idx)[0]

                for child in children_indices:
                    self.parents[child] = parent_idx
                    for f in range(self.length):
                        self.local_t[f, child] = np.matmul(self.local_t[f, idx], self.local_t[f, child])

                indices = np.where(self.parents > idx)
                for id in indices:
                    self.parents[id] = self.parents[id] - 1

                self.joints = np.delete(self.joints, idx)
                self.parents = np.delete(self.parents, idx)
                self.local_t = np.delete(self.local_t, idx, axis=1)

    def compute_world_transform(self):
        self.world_t = np.zeros((self.length, self.joints.shape[0], 4, 4))
        for f in range(self.length):
            self.world_t[f, 0] = np.eye(4)
            for j in range(0, self.joints.shape[0]):
                local_t = self.local_t[f, j]
                self.world_t[f, j] = np.matmul(self.world_t[f, self.parents[j]], local_t)
    
    # acc 추가
    def compute_ref_transformations(self, ref_idx):
        self.ref_t = self.world_t.copy()
        ref_t_list = []
        for f in range(self.length):
            ref_t = self.ref_t[f, ref_idx].copy()
            ref_t = XZProjection(ref_t)
            ref_t_list.append(ref_t)
            for j in range(self.joints.shape[0]):
                self.ref_t[f, j] = np.matmul(np.linalg.inv(ref_t), self.ref_t[f, j])
            # for j in range(self.ref_imu_a.shape[1]):
            #     self.ref_imu_a[f, j] = np.matmul(np.linalg.inv(ref_t)[:3, :3], self.ref_imu_a[f, j])
        ref_t_array = np.asarray(ref_t_list)
        self.ref_t = np.concatenate((np.expand_dims(ref_t_array, 1), self.ref_t), axis=1)

    def write_csv(self, path, representation, precision):
        self.imu_a = self.imu_a.reshape((self.length, -1))
        if representation == 'local':
            data = self.local_t.reshape((self.length, -1))
        elif representation == 'world':
            data = self.world_t.reshape(self.length, -1)
        elif representation == 'ref':
            data = self.ref_t.reshape(self.length, -1)
        else:
            print("wrong representation")
            return
        print(data.shape)
        print(self.imu_a.shape)
        print(self.contact.shape)
        data = np.concatenate((data, self.imu_a, self.contact), axis=-1)
        np.savetxt(path + str(self.name) + '_' + representation + '.csv', data, delimiter=',', fmt=precision)

if __name__ == '__main__':
    # path = '../data_raw/meta_postprocessed/SEA/S14_190_exp/'
    # path = '../data_raw/env/S14_190_exp/'
    path = '../data_raw/totalcapture_retargeted/'
    filenames = os.listdir(path)
    for idx, filename in enumerate(filenames):
        if filename.endswith('.bvh'):
            name = os.path.splitext(filename)[0]
            print(name)
            a = Animation()
            a.load_bvh(path + name)
            a.load_imu_mvnx(path + name)
            # a.load_env(path + name)
            a.compute_world_transform()
            a.compute_ref_transformations(head_idx)

            a.write_csv(path, 'local',  '%1.6f')
            exit()
            # a.write_csv(path, 'world',  '%1.6f')  
            # a.write_csv(path, 'ref',  '%1.6f')
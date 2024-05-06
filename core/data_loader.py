import time
import numpy as np
import torch
from munch import Munch
from torch.utils import data
        
class MocapIMUDataset(data.Dataset):
    def __init__(self, path, window_size, batch_size):
        self.window_size = window_size
        self.batch_size = batch_size
        dataset = np.load(path, 'r', allow_pickle=True)
        self.data_count = len(dataset['name'])
        self.name = dataset['name']
        self.local_t = dataset['local_t']
        self.world_t = dataset['world_t']
        self.ref_t = dataset['ref_t']
        self.imu_a = dataset['imu_a']
        self.contact = dataset['contact']
        
    def __len__(self):
        return max(self.name.shape[0], self.batch_size)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        np.random.seed(int((time.time() * 100000) % 10000))
        idx = np.random.randint(self.data_count)
        while self.local_t[idx].shape[0] < self.window_size + 2:
            idx = np.random.randint(self.data_count)
        end = np.random.randint(self.window_size + 2, self.local_t[idx].shape[0])
        start = end - self.window_size
                         
        sample = {'local_t': torch.tensor(self.local_t[idx][start:end]).float(),
                  'world_t': torch.tensor(self.world_t[idx][start:end]).float(),
                  'ref_t': torch.tensor(self.ref_t[idx][start:end]).float(),
                  'imu_a': torch.tensor(self.imu_a[idx][start:end]).float(),
                  'contact': torch.tensor(self.contact[idx][start:end]).float()}
        return sample
    
    def get_item(self, idx):
        local_t = np.array(self.local_t[idx], float)
        world_t = np.array(self.world_t[idx], float)
        ref_t = np.array(self.ref_t[idx], float)
        imu_a = np.array(self.imu_a[idx], float)
        contact = np.array(self.contact[idx], float)

        # batch = 1 like
        sample = {'name': self.name[idx],
                  'local_t': torch.tensor(local_t).float().unsqueeze(0),
                  'world_t': torch.tensor(world_t).float().unsqueeze(0),
                  'ref_t': torch.tensor(ref_t).float().unsqueeze(0),
                  'imu_a': torch.tensor(imu_a).float().unsqueeze(0),
                  'contact': torch.tensor(contact).float().unsqueeze(0)}
        return sample
    
class SkelDataset(data.Dataset):
    def __init__(self, path, window_size, batch_size):
        self.frames = [750, 750, 500, 700, 700, 750]
        self.window_size = window_size
        self.batch_size = batch_size
        dataset = np.load(path, 'r', allow_pickle=True)
        self.data_count = len(dataset['name'])
        self.name = dataset['name']
        self.local_t = dataset['local_t']
        self.world_t = dataset['world_t']
        self.ref_t = dataset['ref_t']
        self.imu_a = dataset['imu_a']
        self.contact = dataset['contact']
        
    def __len__(self):
        return max(self.name.shape[0], self.batch_size)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        np.random.seed(int((time.time() * 100000) % 10000))
        idx = np.random.randint(self.data_count)
        
        name = self.name[idx]                       
        local_t = np.array(self.local_t[idx], float)
        world_t = np.array(self.world_t[idx], float)
        ref_t = np.array(self.ref_t[idx], float)
        imu_a = np.array(self.imu_a[idx], float)
        contact = np.array(self.contact[idx], float)
               
        sample = {'local_t': torch.tensor(local_t).float(),
                  'world_t': torch.tensor(world_t).float(),
                  'ref_t': torch.tensor(ref_t).float(),
                  'imu_a': torch.tensor(imu_a).float(),
                  'contact': torch.tensor(contact).float()}
        return sample
    
    def get_item(self, idx):
        local_t = np.array(self.local_t[idx], float)
        world_t = np.array(self.world_t[idx], float)
        ref_t = np.array(self.ref_t[idx], float)
        imu_a = np.array(self.imu_a[idx], float)
        contact = np.array(self.contact[idx], float)
        
        # batch = 1 like
        sample = {'name': self.name[idx],
                  'local_t': torch.tensor(local_t).float().unsqueeze(0),
                  'world_t': torch.tensor(world_t).float().unsqueeze(0),
                  'ref_t': torch.tensor(ref_t).float().unsqueeze(0),
                  'imu_a': torch.tensor(imu_a).float().unsqueeze(0),
                  'contact': torch.tensor(contact).float().unsqueeze(0)}
        return sample
    
    def get_item_by_name(self, name):
        if "_" in name:
            subject = name.split('_')[0]
        if "-" in name:
            subject = name.split('-')[0]
        idx = np.where(self.name == subject)[0][0]
        local_t = np.array(self.local_t[idx], float)
        world_t = np.array(self.world_t[idx], float)
        ref_t = np.array(self.ref_t[idx], float)
        imu_a = np.array(self.imu_a[idx], float)
        contact = np.array(self.contact[idx], float)
        
        # batch = 1 like
        sample = {'name': self.name[idx],
                  'local_t': torch.tensor(local_t).float().unsqueeze(0),
                  'world_t': torch.tensor(world_t).float().unsqueeze(0),
                  'ref_t': torch.tensor(ref_t).float().unsqueeze(0),
                  'imu_a': torch.tensor(imu_a).float().unsqueeze(0),
                  'contact': torch.tensor(contact).float().unsqueeze(0)}
        return sample
    
class MocapIMUEnvDataset(data.Dataset):
    def __init__(self, path, window_size, batch_size):
        self.window_size = window_size
        self.batch_size = batch_size
        dataset = np.load(path, 'r', allow_pickle=True)
        self.data_count = len(dataset['name'])
        self.name = dataset['name']
        self.local_t = dataset['local_t']
        self.world_t = dataset['world_t']
        self.ref_t = dataset['ref_t']
        self.imu_a = dataset['imu_a']
        self.contact = dataset['contact']
        self.env = dataset['env']
        
    def __len__(self):
        return max(self.name.shape[0], self.batch_size)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        np.random.seed(int((time.time() * 100000) % 10000))
        idx = np.random.randint(self.data_count)
        end = np.random.randint(self.window_size + 2, self.local_t[idx].shape[0])
        start = end - self.window_size
                         
        sample = {'local_t': torch.tensor(self.local_t[idx][start:end]).float(),
                  'world_t': torch.tensor(self.world_t[idx][start:end]).float(),
                  'ref_t': torch.tensor(self.ref_t[idx][start:end]).float(),
                  'imu_a': torch.tensor(self.imu_a[idx][start:end]).float(),
                  'contact': torch.tensor(self.contact[idx][start:end]).float(),
                  'env': torch.tensor(self.env[idx][start:end]).float()}
        return sample
    
    def get_item(self, idx):
        local_t = np.array(self.local_t[idx], float)
        world_t = np.array(self.world_t[idx], float)
        ref_t = np.array(self.ref_t[idx], float)
        imu_a = np.array(self.imu_a[idx], float)
        contact = np.array(self.contact[idx], float)
        env = np.array(self.env[idx], float)
        
        # batch = 1 like
        sample = {'name': self.name[idx],
                  'local_t': torch.tensor(local_t).float().unsqueeze(0),
                  'world_t': torch.tensor(world_t).float().unsqueeze(0),
                  'ref_t': torch.tensor(ref_t).float().unsqueeze(0),
                  'imu_a': torch.tensor(imu_a).float().unsqueeze(0),
                  'contact': torch.tensor(contact).float().unsqueeze(0),
                  'env': torch.tensor(env).float().unsqueeze(0)}
        return sample
    
def get_dataloader(target, path, window_size, batch_size, num_workers=0):
    path = "data_npz/" + path
    print('Preparing DataLoader for ' + path + ' dataset...')
    if target == 'skel':
        dataset = SkelDataset(path, window_size, batch_size)
    elif target == 'pose':
        dataset = MocapIMUDataset(path, window_size, batch_size)
    elif target == 'env':
        dataset = MocapIMUEnvDataset(path, window_size, batch_size)
    
    # print(path)
    # print(dataset.data_count)
    # for i in range(dataset.data_count):
    #      print(dataset.name[i])
    
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=False,
                           drop_last=True,
                           shuffle=True)
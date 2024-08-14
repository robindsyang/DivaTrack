import os
import re
import random
import numpy as np
import time
import preproc.animation as anim
import argparse

head_idx = 6
window_size = 60

def main(args):
    file_list = []
    print(args.dataset_name + " start building npz.")
    for (root, dirs, files) in os.walk(args.filepath):
        for f in files:
            if f.endswith(".bvh"):
                file_list.append(os.path.join(root, os.path.splitext(f)[0]))

    random.shuffle(file_list)
    train_list = file_list[:int(len(file_list)*0.8)]
    valid_list = file_list[int(len(file_list)*0.8):]

    if args.test == True:
        train_list = train_list[:8]
        valid_list = valid_list[:2]

    sets = [train_list, valid_list]
    set_names = ['train', 'valid']

    if args.split == "False":
        print("single set")    
        sets = [file_list]
        set_names = ['test']

    for i, set in enumerate(sets):
        name_list = []
        local_t_list = []
        world_t_list = []
        ref_t_list = []
        imu_a_list = []
        contact_list = []
        for name in set:
            start_time = time.time()
            print(name + " start")
            data = anim.Animation()
            data.load_bvh(name)
            data.load_imu_mvnx(name)
            data.compute_world_transform()
            data.compute_ref_transformations(head_idx)
            
            name_list.append(data.name)
            local_t_list.append(data.local_t)
            world_t_list.append(data.world_t)
            ref_t_list.append(data.ref_t)
            imu_a_list.append(data.imu_a)
            contact_list.append(data.contact)    
            end_time = time.time()
            print(data.name + ' done / elapsed_time = ' + str(end_time - start_time))
            
        name_arr = np.array(name_list, dtype=str)
        local_t_arr = np.array(local_t_list, dtype=object)
        world_t_arr = np.array(world_t_list, dtype=object)
        ref_t_arr = np.array(ref_t_list, dtype=object)
        imu_a_arr = np.array(imu_a_list, dtype=object)
        contact_arr = np.array(contact_list, dtype=object)

        np.savez_compressed(args.outpath + args.dataset_name + "_{}.npz".format(set_names[i]),
                            name=name_arr,
                            local_t=local_t_arr,
                            world_t=world_t_arr,
                            ref_t=ref_t_arr,
                            imu_a = imu_a_arr,
                            contact = contact_arr)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # arguments
    parser.add_argument('--filepath', type=str, default='data_raw/',
                        help='raw data (.bvh, .mvnx) file path')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='name of .npz file')
    parser.add_argument('--outpath', type=str, default='data_npz/',
                        help='outpath')
    parser.add_argument('--test', type=bool, default=False, help='testset')
    parser.add_argument('--split', type=str, default="True", help='split')
    
    args = parser.parse_args()
    main(args)
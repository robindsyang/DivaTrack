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

    # train 8 : test 2
    random.shuffle(file_list)
    train_list = file_list[:int(len(file_list)*0.8)]
    valid_list = file_list[int(len(file_list)*0.8):]

    if args.test == True:
        train_list = train_list[:8]
        valid_list = valid_list[:2]

    sets = [train_list, valid_list]
    set_names = ['train', 'valid']

    print(args.split)    
    if args.split == "False":
        print("single set")    
        sets = [file_list]
        set_names = ['test']
    
    # static_list = []
    # locomotion_list = []
    # object_list = []
    # sets = [static_list, locomotion_list, object_list]
    # set_names = ['static', 'locomotion', 'object']

    # for i in range(len(file_list)):
    #     path = os.path.split(file_list[i])[-1]
    #     anim_num = re.findall(r'0\d{2}', path)
    #     anim_num = int(anim_num[-1])
    #     print(anim_num)

    #     if anim_num < 25:
    #         static_list.append(file_list[i])
    #     elif anim_num < 34 and anim_num > 24:
    #         locomotion_list.append(file_list[i])
    #     else:
    #         object_list.append(file_list[i])

    # print(static_list)
    # print("--------------------------------")
    # print(locomotion_list)
    # print("--------------------------------")
    # print(object_list)
    # exit()

    for i, set in enumerate(sets):
        name_list = []
        local_t_list = []
        world_t_list = []
        ref_t_list = []
        # ref_traj_list = []
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
            
            # chunks
            """
            split by window_size, overlap
            represence with current reference transformation (ref_inv * world)
            1. transformation -> per frame
            2. trajectory -> w.r.t. current
            3. IMU -> per frame
            
            current_frame = window_size
            while current_frame < data.length:        
                start_time = time.time()
                print(name + "_" + str(current_frame) + " start")
                local_t = data.local_t[current_frame-window_size:current_frame]
                world_t = data.world_t[current_frame-window_size:current_frame]
                ref_t = data.ref_t[current_frame-window_size:current_frame]
                ref_imu_a = data.ref_imu_a[current_frame-window_size:current_frame]
                contact = data.contact[current_frame-window_size:current_frame]
                
                current_ref_t = ref_t[-1, 0]
                ref_traj = data.world_t[current_frame-window_size:current_frame].copy()
                ref_traj[:, :] = np.linalg.inv(current_ref_t) @ ref_traj[:, :]
                ref_traj = np.concatenate((np.tile(current_ref_t, (window_size, 1, 1, 1)), ref_traj), axis=1)
                
                name_list.append(data.name + "_" + str(current_frame))
                local_t_list.append(local_t)
                world_t_list.append(world_t)
                ref_t_list.append(ref_t)
                ref_traj_list.append(ref_traj)
                ref_imu_a_list.append(ref_imu_a)
                contact_list.append(contact)    
                end_time = time.time()
                print(data.name + "_" + str(current_frame) + ' done / elapsed_time = ' + str(end_time - start_time))
                current_frame = current_frame + window_size // 2
            """
            
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
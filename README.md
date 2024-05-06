# DivaTrack

# docker
docker pull robindsyang/divatrack

# dataset link
http://naver.me/FwSksnVJ

# build_dataset
python build_dataset.py --filepath $path_to_raw_bvh&mvnx_data --dataset_name $npz_name --outpath $path_to_save_npz

--test True/False -> use only 10 clips for debugging or use entire set

--split True/False -> to split traing/test set or generate a single .npz

# infer
python main.py --mode infer --model DTVel --target pose --valid_set $inference_dataset_name.npz --cp_datetime 0510_063301 --resume_iter 60000

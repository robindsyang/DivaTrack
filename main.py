import argparse
from munch import Munch
from core.data_loader import get_dataloader
from model.DivaTrackVel import DivaTrackVel
from model.FC_calib import FC_calib

def main(args):
    print(args)

    if args.mode == 'infer' or args.mode == 'export':
        loaders = Munch(valid=get_dataloader(target=args.target, path=args.valid_set, window_size=args.window_size, batch_size=args.batch_size),
                        skel=get_dataloader(target='skel', path="meta22skel_test.npz", window_size=args.window_size, batch_size=args.batch_size))
        
    model = DivaTrackVel(args, loaders)
    
    if args.mode == 'train':
        model.train()
    elif args.mode == 'infer':
        model.infer()
    elif args.mode == 'export':
        model.export()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
        
    # model parameters
    parser.add_argument('--model', type=str, default='tcvae_auto_9d')
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--hid_dim', type=int, default=1024)
    parser.add_argument('--lat_dim', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--mask_ratio', type=float, default=0.9)
    parser.add_argument('--cmask_ratio', type=float, default=0.1)

    # hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sc_lr', type=float, default=1e-5)
    parser.add_argument('--sc_wd', type=float, default=1e-6)
    parser.add_argument('--p_lr', type=float, default=1e-4)
    parser.add_argument('--p_wd', type=float, default=1e-6)
    parser.add_argument('--b_lr', type=float, default=1e-4)
    parser.add_argument('--lambda_pose', type=float, default=1.0)
    parser.add_argument('--lambda_vel', type=float, default=1e-1)
    parser.add_argument('--lambda_cont', type=float, default=1e-3)
    parser.add_argument('--lambda_consist', type=float, default=1e-5)
    parser.add_argument('--lambda_fk', type=float, default=1e-1)
    parser.add_argument('--lambda_kl', type=float, default=1e-3)

    # training arguments 
    parser.add_argument('--scheduler', type=int, default = 0)
    parser.add_argument('--resume_iter', type=int, default=100000)
    parser.add_argument('--total_iter', type=int, default=100000)
    parser.add_argument('--warmup_iter', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=256)

    # misc
    parser.add_argument('--target', type=str, default = 'pose', choices=['skel', 'pose', 'env'])
    parser.add_argument('--mode', type=str, default = 'train', choices=['train', 'infer', 'export'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # directories for training
    parser.add_argument('--train_set', type=str, default='test_train.npz')
    parser.add_argument('--valid_set', type=str, default='test_valid.npz')
    parser.add_argument('--tb_dir', type=str, default='expr/runs')
    parser.add_argument('--tb_comment', type=str, default='test')
    parser.add_argument('--cp_dir', type=str, default='expr/checkpoints')
    parser.add_argument('--onnx_dir', type=str, default='expr/onnx')
    parser.add_argument('--cp_datetime', type=str, default='')

    # directories for inference
    parser.add_argument('--infer_set', type=str, default='valid')
    parser.add_argument('--result_dir', type=str, default='expr/results')
    parser.add_argument('--provide_gtcont', type=str, default='False')
    parser.add_argument('--provide_gtskel', type=str, default='False')

    # debug
    parser.add_argument('--graph_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--partial', type=str, default='False')

    args = parser.parse_args()
    main(args)
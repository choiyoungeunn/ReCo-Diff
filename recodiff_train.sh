#!/usr/bin/env bash

epochs=20  # Number of epochs
dataset_shape=256  # CT image size (squared)
res_dir="/media/silveryong/CCD2E259D2E246F4/silveryong/code/ReCo-Diff/logs"  # enter your directory for storing result
dataset_path='/media/silveryong/CCD2E259D2E246F4/silveryong/code/CvG-Diff-main/preprocess_data/aapm16/train_img' # enter your dataset path of train images

network='ReCo-Diff'
tb_root="$res_dir/$network/tensorboard"

# Resume settings (set paths to enable):
# 'resume=false': do NOT load network weights from checkpoint.
# 'resume_opt=false': do NOT load optimizer/scheduler states from checkpoint.
# 'net_checkpath' and 'opt_checkpath' are still passed as CLI arguments,
# but they are only used when '--resume' / '--resume_opt' are enabled.
# With both flags set to false, training starts from freshly initialized states.
resume=false
resume_opt=false
net_checkpath="$res_dir/recodiff/ckpt/$network/${network}-net-colddiff_ema_0_epoch.pkl"
opt_checkpath="$res_dir/recodiff/ckpt/$network/${network}-opt-colddiff_latest_epoch.pkl"

# Error-conditioned CFG settings
err_cfg_sigma=1.0

CUDA_VISIBLE_DEVICES="0" python colddiff_main.py --epochs $epochs \
--lr 4e-5 --optimizer 'adam' \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--network $network \
--loss 'l2' --trainer_mode 'train' \
--checkpoint_root $res_dir'/'$network'/ckpt' \
--checkpoint_dir $network \
--dataset_path $dataset_path \
--batch_size 4 --num_workers 4 --log_interval 200 \
--use_tqdm \
--use_tensorboard \
--tensorboard_root "$tb_root" \
--tensorboard_dir $network \
$( [ "$resume" = true ] && echo --resume ) \
$( [ "$resume_opt" = true ] && echo --resume_opt ) \
--net_checkpath "$net_checkpath" \
--opt_checkpath "$opt_checkpath" \
--use_err_cfg \
--err_cfg_sigma $err_cfg_sigma \

# If you want to use Weights & Biases (wandb), just use the lines below:
# --use_wandb --run_name $network \
# --wandb_root $res_dir'/cvgdiff/wandb' \
# --wandb_dir $network \

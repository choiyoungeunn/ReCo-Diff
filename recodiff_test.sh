dataset_shape=256  # CT image size (squared)
res_dir='/path/to/your/ReCo-Diff/logs' # enter your directory that stores trained models
dataset_path='/path/to/your/preprocess_data/aapm16/test_img' # enter your dataset path of test images
network='ReCo-Diff'
net_checkpath_default='/path/to/your/ReCo-Diff/logs/ReCo-Diff/ckpt/ReCo-Diff/ReCo-Diff-net-colddiff_best_epoch.pkl'
net_checkpath="${NET_CHECKPATH:-$net_checkpath_default}"
pkl_base="$(basename "$net_checkpath" .pkl)"

#view
view=18
CUDA_VISIBLE_DEVICES="0" python colddiff_main.py \
--num_views $view \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--network $network \
--trainer_mode 'test' \
--split 'test' \
--dataset_path $dataset_path \
--use_err_cfg \
--no_iterative_sampling \
--net_checkpath "$net_checkpath" \
--tester_save_path $res_dir \
--tester_save_name $network'/results/test_'$view'v-'$pkl_base \
--tester_save_image \


view=36
CUDA_VISIBLE_DEVICES="0" python colddiff_main.py \
--num_views $view \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--network $network \
--trainer_mode 'test' \
--split 'test' \
--dataset_path $dataset_path \
--use_err_cfg \
--no_iterative_sampling \
--net_checkpath "$net_checkpath" \
--tester_save_path $res_dir \
--tester_save_name $network'/results/test_'$view'v-'$pkl_base \
--tester_save_image \


view=72
CUDA_VISIBLE_DEVICES="0" python colddiff_main.py \
--num_views $view \
--dataset_name 'aapm' --dataset_shape $dataset_shape \
--network $network \
--trainer_mode 'test' \
--split 'test' \
--dataset_path $dataset_path \
--use_err_cfg \
--no_iterative_sampling \
--net_checkpath "$net_checkpath" \
--tester_save_path $res_dir \
--tester_save_name $network'/results/test_'$view'v-'$pkl_base \
--tester_save_image \

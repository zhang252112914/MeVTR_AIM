CUDA_VISIBLE_DEVICES=1,3,6,7 torchrun --nproc_per_node=4 \
--master_port=53198 \
main_task_retrieval_multi.py --do_train \
--num_thread_reader=2 \
--data_path /home/zyl/MeVTR_data_and_models/charades/annotation \
--features_path /home/zyl/MeVTR_data_and_models/charades/Charades_v2_3 \
--output_dir /home/zyl/MeVTR/output  \
--max_words 77 --max_frames 64 --batch_size_val 64 \
--datatype charades --feature_framerate 1 --coef_lr 1e-4 \
--slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--pretrained_clip_name ViT-B/32 \
--post_process cluster --post_cluster_centroids 16 \
--regularize none
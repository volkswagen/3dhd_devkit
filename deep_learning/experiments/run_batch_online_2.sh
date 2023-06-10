# Type 'bash run_batch_2.sh' in terminal to execute
python run_training.py --experiment "dd-s3-28m" --dd_stage 3 || pkill python
## OBJECT DETECTION ###
## Single task object detection for {lights, poles, signs} 60m
#python run_training.py --experiment "ed-l-3dhdnet-60m" --configured_element_types lights --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
#python run_training.py --experiment "ed-p-3dhdnet-60m" --configured_element_types poles --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
#python run_training.py --experiment "ed-s-3dhdnet-60m" --configured_element_types signs --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
##
### Multitask object detection with {3dhdnet, pointpillars, voxelnet} 60m
#python run_training.py --experiment "ed-lps-pointpillars-60m" --use_amp False --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --voxel_size .2 .2 9.6 --min_points_per_voxel 1 --max_points_per_voxel 100 --vfe_class_name "PillarFeatureNet" --vfe_num_filters 64 --middle_class_name "PointPillarsScatter" --bb_num_upsample_filters 256 256 256 --use_3d_backbone False --use_3d_heads False --set_anchor_layers_like_voxels False || pkill python
#python run_training.py --experiment "ed-lps-voxelnet-60m" --use_amp False --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --voxel_size .2 .2 .4 --min_points_per_voxel 1 --max_points_per_voxel 24 --vfe_class_name 'VoxelFeatureExtractor' --vfe_num_filters 32 128 --middle_class_name 'MiddleExtractor' --bb_num_filters 128 128 256 --bb_num_upsample_filters 256 256 256 --use_3d_backbone False --use_3d_heads False || pkill python
#python run_training.py --experiment "ed-lps-3dhdnet-60m" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
##
## DEVIATION DETECTION ###
#python run_training.py --experiment "dd-s3-train-2-2-1" --dd_stage 3 --deletion_prob 0.02 --insertion_prob 0.02 --substitution_prob 0.01 --load_lvl_hdpc online --load_lvl_map online --num_workers 8 || pkill python
#python run_training.py --experiment "dd-s3-train-10-10-5" --dd_stage 3 --deletion_prob 0.10 --insertion_prob 0.10 --substitution_prob 0.05 --load_lvl_hdpc online --load_lvl_map online --num_workers 8 || pkill python
#python run_training.py --experiment "dd-s3-train-20-20-10" --dd_stage 3 --deletion_prob 0.20 --insertion_prob 0.20 --substitution_prob 0.10 --load_lvl_hdpc online --load_lvl_map online --num_workers 8 || pkill python
#python run_training.py --experiment "dd-s3-train-30-30-15" --dd_stage 3 --deletion_prob 0.30 --insertion_prob 0.30 --substitution_prob 0.15 --load_lvl_hdpc online --load_lvl_map online --num_workers 8 || pkill python
#python run_training.py --experiment "dd-s3-60m-train-20-20-10" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --dd_stage 3 --deletion_prob 0.20 --insertion_prob 0.20 --substitution_prob 0.10 --load_lvl_hdpc online --load_lvl_map online --num_workers 8 || pkill python
#python run_training.py --experiment "dd-s3-60m-train-10-10-5" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --dd_stage 3 --deletion_prob 0.10 --insertion_prob 0.10 --substitution_prob 0.05 --load_lvl_hdpc online --load_lvl_map online --num_workers 8 || pkill python
#python run_training.py --experiment "dd-s3-60m-train-30-30-15" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --dd_stage 3 --deletion_prob 0.30 --insertion_prob 0.30 --substitution_prob 0.15 --load_lvl_hdpc online --load_lvl_map online --num_workers 8 || pkill python
python run_training.py --experiment "id" --dd_stage 3 --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --load_lvl_hdpc online --load_lvl_map online --num_workers 8 || pkill python
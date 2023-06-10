## OBJECT DETECTION ###
python run_training.py --experiment "dd-s3-60m-online-2.0" --dd_stage 3 --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --load_lvl_hdpc online --load_lvl_map online --num_workers 8 || pkill python
## Single task object detection for {lights, poles, signs} 60m
python run_training.py --experiment "od-l-3dhdnet-60m" --configured_element_types lights --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
python run_training.py --experiment "od-p-3dhdnet-60m" --configured_element_types poles --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
python run_training.py --experiment "od-s-3dhdnet-60m" --configured_element_types signs --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
##
### Multitask object detection with {3dhdnet, pointpillars, voxelnet} 60m
#python run_training.py --experiment "od-lps-3dhdnet-60m" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
#python run_training.py --experiment "od-lps-pointpillars-60m" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --voxel_size .2 .2 9.6 --min_points_per_voxel 1 --max_points_per_voxel 100 --vfe_class_name "PillarFeatureNet" --vfe_num_filters 64 --middle_class_name "PointPillarsScatter" --bb_num_upsample_filters 256 256 256 --use_3d_backbone False --use_3d_heads False --set_anchor_layers_like_voxels False || pkill python
#python run_training.py --experiment "od-lps-voxelnet-60m" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --voxel_size .2 .2 .4 --min_points_per_voxel 1 --max_points_per_voxel 24 --vfe_class_name 'VoxelFeatureExtractor' --vfe_num_filters 32 128 --middle_class_name 'MiddleExtractor' --bb_num_filters 128 128 256 --bb_num_upsample_filters 256 256 256 --use_3d_backbone False --use_3d_heads False || pkill python
##
### DEVIATION DETECTION ###
### Default runs stage {1, 2, 3} 60m (1: MDD-SC, 2: MDD-MC, 3: MDD-M)
#python run_training.py --experiment "dd-s3-60m-generated-2.0" --dd_stage 3 --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
#python run_training.py --experiment "dd-s2-60m-generated-2.0" --dd_stage 2 --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
#python run_training.py --experiment "dd-s1-60m-generated-2.0" --dd_stage 1 --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
##
### Default runs stage {1, 2, 3} 30m
#python run_training.py --experiment "dd-s3-30m" --dd_stage 3 --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s2-30m" --dd_stage 2 --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s1-30m" --dd_stage 1 --any_elements_per_sample_min 1 || pkill python
##
### Point density
#python run_training.py --experiment "dd-s3-30m-pd-50" --dd_stage 3 --dd_point_density 0.50 --point_dropout False --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s3-30m-pd-25" --dd_stage 3 --dd_point_density 0.25 --point_dropout False --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s3-30m-pd-10" --dd_stage 3 --dd_point_density 0.10 --point_dropout False --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s2-30m-pd-50" --dd_stage 2 --dd_point_density 0.50 --point_dropout False --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s2-30m-pd-25" --dd_stage 2 --dd_point_density 0.25 --point_dropout False --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s2-30m-pd-10" --dd_stage 2 --dd_point_density 0.10 --point_dropout False --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s1-30m-pd-50" --dd_stage 1 --dd_point_density 0.50 --point_dropout False --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s1-30m-pd-25" --dd_stage 1 --dd_point_density 0.25 --point_dropout False --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s1-30m-pd-10" --dd_stage 1 --dd_point_density 0.10 --point_dropout False --any_elements_per_sample_min 1 || pkill python
##
### Occlusion ablation study
#python run_training.py --experiment "dd-s3-30m-occ-25" --dd_stage 3 --occlusion_prob 0.25 --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s3-30m-occ-50" --dd_stage 3 --occlusion_prob 0.50 --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s3-30m-occ-75" --dd_stage 3 --occlusion_prob 0.75 --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s2-30m-occ-25" --dd_stage 2 --occlusion_prob 0.25 --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s2-30m-occ-50" --dd_stage 2 --occlusion_prob 0.50 --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s2-30m-occ-75" --dd_stage 2 --occlusion_prob 0.75 --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s1-30m-occ-25" --dd_stage 1 --occlusion_prob 0.25 --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s1-30m-occ-50" --dd_stage 1 --occlusion_prob 0.50 --any_elements_per_sample_min 1 || pkill python
#python run_training.py --experiment "dd-s1-30m-occ-75" --dd_stage 1 --occlusion_prob 0.75 --any_elements_per_sample_min 1 || pkill python
## Idle
python run_training.py --experiment "id" --num_epochs 100 --dd_stage 3 --batch_size 1 --fm_extent -10 60.4 -20 20 -2 7.6 || pkill python
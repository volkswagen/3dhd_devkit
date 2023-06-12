# Type 'bash run_batch.sh' in terminal to execute
## 3D ELEMENT RECOGNITION ###
## Multitask element recognition
python run_training.py --experiment "er-lps-3dhdnet-60m" --classification_active True --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
## Single task element detection for {lights, poles, signs} 60m
python run_training.py --experiment "ed-l-3dhdnet-60m" --configured_element_types lights --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
python run_training.py --experiment "ed-p-3dhdnet-60m" --configured_element_types poles --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
python run_training.py --experiment "ed-s-3dhdnet-60m" --configured_element_types signs --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
## Multitask element detection with {3DHDNet, PointPilalrs, VoxelNet} 60m
python run_training.py --experiment "ed-lps-3dhdnet-60m" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
python run_training.py --experiment "ed-lps-pointpillars-60m" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --voxel_size .2 .2 9.6 --min_points_per_voxel 1 --max_points_per_voxel 100 --vfe_class_name "PillarFeatureNet" --vfe_num_filters 64 --middle_class_name "PointPillarsScatter" --bb_num_upsample_filters 256 256 256 --use_3d_backbone False --use_3d_heads False --set_anchor_layers_like_voxels False --use_amp False || pkill python
python run_training.py --experiment "ed-lps-voxelnet-60m" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --voxel_size .2 .2 .4 --min_points_per_voxel 1 --max_points_per_voxel 24 --vfe_class_name 'VoxelFeatureExtractor' --vfe_num_filters 32 128 --middle_class_name 'MiddleExtractor' --bb_num_filters 128 128 256 --bb_num_upsample_filters 256 256 256 --use_3d_backbone False --use_3d_heads False --use_amp False || pkill python
##
## DEVIATION DETECTION ##
## Default runs stage {1, 2, 3} 60m (1: MDD-SC, 2: MDD-MC, 3: MDD-M)
python run_training.py --experiment "dd-s3-60m" --dd_stage 3 --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
python run_training.py --experiment "dd-s2-60m" --dd_stage 2 --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
python run_training.py --experiment "dd-s1-60m" --dd_stage 1 --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
##
## Point density ablation study
python run_training.py --experiment "dd-s3-30m-pd-50" --dd_stage 3 --dd_point_density 0.50 --point_dropout False --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s3-30m-pd-25" --dd_stage 3 --dd_point_density 0.25 --point_dropout False --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s3-30m-pd-10" --dd_stage 3 --dd_point_density 0.10 --point_dropout False --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s2-30m-pd-50" --dd_stage 2 --dd_point_density 0.50 --point_dropout False --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s2-30m-pd-25" --dd_stage 2 --dd_point_density 0.25 --point_dropout False --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s2-30m-pd-10" --dd_stage 2 --dd_point_density 0.10 --point_dropout False --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s1-30m-pd-50" --dd_stage 1 --dd_point_density 0.50 --point_dropout False --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s1-30m-pd-25" --dd_stage 1 --dd_point_density 0.25 --point_dropout False --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s1-30m-pd-10" --dd_stage 1 --dd_point_density 0.10 --point_dropout False --any_elements_per_sample_min 1 || pkill python
##
## Occlusion ablation study
python run_training.py --experiment "dd-s3-30m-occ-25" --dd_stage 3 --occlusion_prob 0.25 --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s3-30m-occ-50" --dd_stage 3 --occlusion_prob 0.50 --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s3-30m-occ-75" --dd_stage 3 --occlusion_prob 0.75 --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s2-30m-occ-25" --dd_stage 2 --occlusion_prob 0.25 --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s2-30m-occ-50" --dd_stage 2 --occlusion_prob 0.50 --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s2-30m-occ-75" --dd_stage 2 --occlusion_prob 0.75 --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s1-30m-occ-25" --dd_stage 1 --occlusion_prob 0.25 --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s1-30m-occ-50" --dd_stage 1 --occlusion_prob 0.50 --any_elements_per_sample_min 1 || pkill python
python run_training.py --experiment "dd-s1-30m-occ-75" --dd_stage 1 --occlusion_prob 0.75 --any_elements_per_sample_min 1 || pkill python
## Idle
python run_training.py --experiment "id" --num_epochs 100 --dd_stage 3 --batch_size 1 --fm_extent -10 60.4 -20 20 -2 7.6 || pkill python
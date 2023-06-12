# 3DHD DevKit

![map_fm_d](https://github.com/volkswagen/3dhd_devkit/assets/113357338/b97ed7ce-35c9-4223-97c3-1f72a01e2508)

The 3DHD DevKit comprises our entire deep learning pipeline
for LiDAR-based **3D map element recognition** and **map deviation detection**.
The DevKit has been created by [Christopher Plachetka](https://www.linkedin.com/in/christopher-plachetka-42b325115/)
(PhD candidate in his final year) and [Benjamin Sertolli](https://www.linkedin.com/in/sertolli/?originalSubdomain=de) (former master student). 

Subsequently, we provide instructions on how to operate 

1. our **3D map element recognition** pipeline detecting and classifying traffic lights, traffic signs, and poles,
2. our **map deviation detection** pipeline, 
3. and our **visualization framework** for displaying predictions, point clouds, and map data.

Our dataset 3DHD CityScenes can be downloaded [here](https://zenodo.org/record/7085090#.ZH2V8WfP1aQ). 

## Citation

When using our DevKit, you are welcome to cite [[Link](https://www.researchgate.net/publication/368983255_DNN-Based_Map_Deviation_Detection_in_LiDAR_Point_Clouds)]:
```
@article{Plachetka.2023,
   author={Plachetka, Christopher and Sertolli, Benjamin and Fricke, Jenny and Klingner, Marvin and Fingscheidt, Tim},
   year = {2023},
   title = {DNN-Based Map Deviation Detection in LiDAR Point Clouds},
   pages = {1--23},
   pagination = {page},
   journal = {IEEE Open Journal of Intelligent Transportation Systems (preprint)}}
```

If you use our dataset, please cite [[Link](https://www.researchgate.net/publication/364309881_3DHD_CityScenes_High-Definition_Maps_in_High-Density_Point_Clouds)]:
```
@INPROCEEDINGS{9921866,
   author={Plachetka, Christopher and Sertolli, Benjamin and Fricke, Jenny and Klingner, Marvin and Fingscheidt, Tim},
   booktitle={2022 IEEE 25th International Conference on Intelligent Transportation Systems (ITSC)}, 
   title={3DHD CityScenes: High-Definition Maps in High-Density Point Clouds}, 
   year={2022},
   pages={627-634}}
```

## Requirements

* Python 3.7
* PyTorch 1.6
* CUDA 10.1
* Recommended: 2-GPU setup with min. 24 GB VRAM (e.g., Tesla V100 32 GB)
* RAM requirements depend on the number of utilized CPU workers. For 8 workers (default): 40 GB  

Newer versions than Python 3.7 have not been tested. 
Also GPUs with 16 GB VRAM can be employed by using a batch size of 1 per GPU and the small default crop 
extent of 28 m in x-dimension.

## Installation

**1. Install all packages found in requirements.txt.** 

Mind the installation order. For our visualization framework, first install VTK, then Mayavi, and finally PySide2.

We recommend installing PyTorch 1.6 using: 
```
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

**2. Download [3DHD CityScenes](https://zenodo.org/record/7085090#.ZH2V8WfP1aQ).** 

We recommend using an SSD as storage to reduce training time.

**3. Adjust utility.system_paths.py according to your system.**

To this end, ```change get_known_systems()```. You can add multiple systems, e.g., for running on a GPU cluster 
or workstation. The argument ```description``` can be chosen freely. Identify your ```hostname``` by running ```python system_paths.py```,
and add your system to the list with the respective hostname. 
The create and set a ```log_dir```, specifying where the DevKit saves and finds experiments.
The ```root_dir``` points to the directory of 3DHD CityScenes. 

```
def get_known_systems():
""" Returns list of all known systems (in SystemBasePaths format). If this project is run on a new system, make
sure to configure the basic paths here first.
"""
known_systems = [  # add your own system here
SystemBasePaths(description='Windows example',
          hostname='example-hostname',
          log_dir=Path(r"C:\example\path\to\logs"),
          root_dir=Path(r"C:\example\path\to\3DHD_CityScenes")
          ),
SystemBasePaths(description='Linux example',
          hostname='example-hostname',
          log_dir=Path("/example/path/to/logs"),
          root_dir=Path("/example/path/to/3DHD_CityScenes")
          ),
]
return known_systems
```

**4. Optional: generate the dataset.** 

To reduce training time to approx. 40 %, we recommend generating point cloud and map crops prior to training. 
Per default, our DevKit generates respective crops in an online fashion without prior generation, which is controlled by ```load_lvl_hdpc``` and ```load_lvl_map``` being set to ```online```
in ```deep_learning.configurations.default_config.ini```. The generated dataset takes up 1.2 TB of disk space.

To generate the dataset:
```
cd path/to/3dhd_devkit/dataset

python hdpc_generator.py
python map_generator.py
```

Generating the point cloud crops takes around 12 hours, while the map crops are generated within approx. 1 hour. 
To utilize the generated dataset, set the settings mentioned above to ```generated```.

## Logging

Logging to TensorBoard and MLflow is active by default (see ```tracking_backend``` in ```deep_learning.configurations.default_config.ini```.
To show experimental logs (losses, metrics, etc.) using TensorBoard, execute:
```
tensorboard --logdir=/logs --samples_per_plugin images=10000 --reload_multifile True
```
With your browser, open ["http://localhost:6006/#"](http://localhost:6006/#) to view the logs. 

For MLflow, adjust the path ```file:/logs/mlruns``` according to your logging directory and execute:
```
mlflow ui --backend-store-uri "file:/logs/mlruns"
```
Next, open http://127.0.0.1:5000 with your browser. 
Note that the ```mlruns``` folder is created automatically during training in your specified logging directory
(set in ```system_paths.py```). 

## Known issues 

* When using VoxelNet or PointPillars as network architectures with automatic mixed precision (AMP, ```use_amp=True``` in ```default_config.ini```)
training stops with an error after a while. Thus, AMP must be disabled in those cases.

* Sparse convolutions as provided by [spconv](https://github.com/traveller59/spconv) do not work with PyTorch's distributed data parallel (DDP, our way to utilize multiple GPUs).

## Disclaimer 

Our code is indented to provide transparency regarding achieved research results and to facilitate further research in this field, serving as a proof of concept only, without being part of or being deployed in any VW products. 
Note that this repository is not maintained. To the best of the author's knowledge, the code is free from errors or relevant security vulnerabilities. However, any possible residual bugs, security vulnerabilities, or any other arising issues from using our code in your application or research, are left to the user’s responsibility to handle. 
This notification is meant to create user expectations appropriate to code developed as proof of concept for research purposes and does not serve as a legal disclaimer. Regarding liability and licensing, note the license file in this repository.

# **1. 3D Map Element Recognition**

![element_recognition_b](https://github.com/volkswagen/3dhd_devkit/assets/113357338/aa6c3aa7-eb0d-499a-bfd5-4084ce4db6a6)

The 3D map element recognition pipeline recognizes (detects and classifies) traffic lights, traffic signs, and poles in LiDAR data.

## 1.1 Performance 

Our applied multitask 3DHDNet achieves the following performances. Note that accuracy is only evaluated for true positive (TP) detections, 
and that single task performance (e.g., only poles) is even higher.

| **Class**  | **F1** | **Recall** | **Precision**  | **Accuracy**
| ------------- | ------------- | ------------- | ------------- | ------------- |
| **Lights (All)**  | 0.92 | 0.91 | 0.93 | 0.91 | 
| People | 0.90  | 0.91 | 0.92 | 0.90 |
| Vehicle | 0.95  | 0.94 | 0.97 | 0.93 |
| Warning | 0.60  | 0.57 | 0.62 | 0.69 | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| **Poles (All)**  | 0.78 | 0.74 | 0.82 | 0.97 | 
| General pole | 0.66 | 0.73 | 0.61 | 0.65 |
| Lamppost | 0.94  | 0.92 | 0.98 | 0.91 |
| Protection pole | 0.71 | 0.65 | 0.78 | 0.99 | 
| Traffic light pole | 0.97 | 0.97 | 0.98 | 0.96 | 
| Traffic sign pole | 0.86 | 0.84 | 0.87 | 0.96 | 
| Tree | 0.86  | 0.85 | 0.87 | 1.0 | 
| ------------- | ------------- | ------------- | ------------- | ------------- |
| **Signs (All)**  | 0.83 | 0.76 | 0.90 | 0.86 | 
| Arrow | 0.81 | 0.77 | 0.86 | 0.77 |
| Circle | 0.84 | 0.76 | 0.94 | 0.85 |
| Diamond | 0.89 | 0.82 | 0.98 | 0.50 | 
| Octagon | 0.93 | 0.87 | 1.00 | 0.00 | 
| Rectangle | 0.79 | 0.73 | 0.87 | 0.91 | 
| Triangle_down | 0.92 | 0.88 | 0.96 | 0.82 | 
| Triangle_up | 0.95 | 0.93 | 0.96 | 0.85 | 

## 1.2 Reproduction

To train the multitask element recognition with 3DHDNet, run:
```
## Multitask element recognition
cd path/to/3dhd_devkit/deep_learning/experiments
python run_training.py --experiment "er-lps-3dhdnet-60m" --classification_active True --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
```

Further, you can obtain plain element detection results without classification, e.g., using single task variants or other architectures, i.e., PointPillars or VoxelNet:
```
## Single task element detection for {lights, poles, signs} 60m
python run_training.py --experiment "ed-l-3dhdnet-60m" --configured_element_types lights --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
python run_training.py --experiment "ed-p-3dhdnet-60m" --configured_element_types poles --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
python run_training.py --experiment "ed-s-3dhdnet-60m" --configured_element_types signs --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
##
## Multitask element detection with {3DHDNet, PointPilalrs, VoxelNet} 60m
python run_training.py --experiment "ed-lps-3dhdnet-60m" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 || pkill python
python run_training.py --experiment "ed-lps-pointpillars-60m" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --voxel_size .2 .2 9.6 --min_points_per_voxel 1 --max_points_per_voxel 100 --vfe_class_name "PillarFeatureNet" --vfe_num_filters 64 --middle_class_name "PointPillarsScatter" --bb_num_upsample_filters 256 256 256 --use_3d_backbone False --use_3d_heads False --set_anchor_layers_like_voxels False --use_amp False || pkill python
python run_training.py --experiment "ed-lps-voxelnet-60m" --batch_size 1 --fm_extent -10 50.8 -20 20 -2 7.6 --voxel_size .2 .2 .4 --min_points_per_voxel 1 --max_points_per_voxel 24 --vfe_class_name 'VoxelFeatureExtractor' --vfe_num_filters 32 128 --middle_class_name 'MiddleExtractor' --bb_num_filters 128 128 256 --bb_num_upsample_filters 256 256 256 --use_3d_backbone False --use_3d_heads False --use_amp False || pkill python
```
Training requires two days using a 60 m extent in x-dimension and a batch size of 1 employing two GPUs (effective batch size being 2).

Alternatively, we provide the trained experiment "er-lps-3dhdnet-60m" in ```_networks```, 
capable of recognizing lights, poles, and signs simultaneously. 
First, unpack the experiment into your ```log``` directory specified in ```system_paths.py```.
To produce the results in Section 1.1, run:
```
cd path/to/3dhd_devkit/deep_learning/experiments

python run_inference.py
python run_evaluation.py
```
Note that per default, inference and evaluation is performed for all published networks. 
You can simply comment out undesired experiments in the respective ```main()``` methods of ```run_inference.py``` and ```run_evaluation```:
```
cfg = {
        'experiments': [
            'dd-s3-60m',
            'dd-s2-60m',
            'dd-s1-60m',
            'er-lps-3dhdnet-60m'
        ],
        'partitions': [
            'val',
            'test'
        ],
        'net_dirs': ['checkpoint']
    }
```

# **2. Map Deviation Detection**

![deviation_detection](https://github.com/volkswagen/3dhd_devkit/assets/113357338/6b32c34a-b266-4c46-8be0-582616389c78)

Our map deviation detection method takes both sensor and map data as input
to detect deviations. 

## 2.1 Map Deviations

Map deviations are simulated during training, while we use a fixed deviation-to-element assignment for validation and test.
Our set of generated deviations can be found in ```_deviations```. 
Copy the JSON files in ```_deviations``` to ```3DHD_CityScenes/Dataset``` (in the ```root_dir```).

To generate your own deviation assignments, execute:
```
cd path/to/3dhd_devkit/dataset

python deviation_exporter.py
```

## 2.2 Performance 

Using our benchmark, our specialized network for map deviation detection (MDD-M, eq. dd-s3-60m in ```_networks```) using the additional map input 
and the classification of evaluation states (VER, DEL, INS, or SUB) achieves: 

| **Method**  | **VER (F1)** | **DEV (F1)** | **E(pos)**  | **E(w)** | **E(d)** | **E(h)** | **E(phi)**
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Lights  | 0.99 | 0.87 | 5.8 cm | 2.3 cm | - | 5.3 cm | 4.4° |
| Poles  | 0.99 | 0.81 | 5.0 cm | - | 2.7 cm | - | - |
| Signs  | 0.99 | 0.87 | 7.3 cm | 5.6 cm | - | 4.8 cm | 4.2° |

VER refers to the detection of verifications (map and sensor data matching), and DEV to deviations, i.e., 
insertions INS (falsely existing elements), deletions DEL (missing elements), and substitutions SUB (interchanged elements). 
Note that when comparing your MDD method to ours, mind to include a comparison for 10 % point density (see below),
which will affect the verification (VER) performance and reveal how well your method handles bad weather or onboard scans. 
For respective ablation study results, see our publication.

## 2.3 Reproduction 

We include the trained networks for MDD-SC, MDD-MC, and MDD-M (see publication) in ```_networks```.
To reproduce our results with the provided networks, unpack these into your ```log``` directory specified in ```system_paths.py```, 
and then execute:
```
cd path/to/3dhd_devkit/deep_learning/experiments

python run_inference.py
python run_evaluation.py
```
The provided experiments are set as defaults in both scripts. 
Change respective settings if inference or evaluation is desired for other experiments. 
After training, both inference and evaluation are called automatically.

For obtaining the precision-recall-curves in our paper, first change ```save_path``` in ```scripts.visualization_scripts```
according to your system, and then execute:
```
cd path/to/3dhd_devkit/scripts

python visualization_scripts.py
```

To train all methods from scratch, and to reproduce our ablation study results regarding point density and occlusion, execute the following code.
Note that S3, S2, S1 equal MDD-M, MDD-MC, and MDD-SC in our paper, respectively.

```
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
```

# 3. Visualization Framework 

First, our visualization framework can visualize predictions obtained from our element recognition or deviation detection pipeline.
Second, the framework provides a dataset viewer for 3DHD CityScenes. 

## 3.1 Sample Viewer

![sample_viewer](https://github.com/volkswagen/3dhd_devkit/assets/113357338/f4c02d83-3107-468e-afc0-3fb5cca69aa0)

Regarding predictions produced by the element recognition pipeline, elements are colored by default according to their evaluation result:
true positive (TP, green), false positive (FP, cyan), false negative (FN, red). Associated ground truth (GT) objects are depicted in gray.
To visualize predictions obtained either from element recognition or deviation detection, use ```visualization.sample_viewer.py```
and make sure ```run_view_predictions()``` is uncommented.


```
def main():
  run_view_predictions()  # visualize network outputs
  # run_view_samples()    # visualize network inputs
````

In ```run_view_predictions()```, you can adjust various settings, e.g., the experiment to load and visualize. 
Depending on the pipeline mode (element recognition or deviation detection), either object detections (signs, lights, or poles)
or object detections with evaluation states are visualized (i.e., VER, DEL, INS, or SUB; see our paper for details). 

When enabling ```run_view_samples()```, only the point cloud and HD map crops as network inputs are visualized.
If the deviation detection pipeline is enabled (```deviation_detection_task=True```, automatically set if ```dd_stage != 0```),
respective deviations are visualized with map and sensor data showing differences. 

To run the sample viewer, execute:

```
cd path/to/3dhd_devkit/visualization

python sample_viewer.py
```

Note that before visualizing predictions, inference must be performed for the desired experiment (```run_inference.py```).

## 3.2 Dataset Viewer

![dataset_viewer](https://github.com/volkswagen/3dhd_devkit/assets/113357338/b02ba42e-ac61-4668-ad2c-96f82491459c)

To plainly visualize the point cloud and HD map data provided by 3DHD CityScenes, 
you can utilize ```visualization.dataset_viewer.py```.
In ```run_view_test_area()```, set data paths according to your system, 
as well as the desired point cloud tile and map elements. Subsequently, run:

```
cd path/to/3dhd_devkit/visualization

python dataset_viewer.py
```

## 3.3 Point Cloud Viewer

![pc_viewer](https://github.com/volkswagen/3dhd_devkit/assets/113357338/26d85cb1-de99-4d0f-82d3-77ee542bf73f)

If you only want to visualize point clouds, we provide ```visualization.point_cloud_viewer.py``` to display point cloud tiles.
In ```run_visualize_binary_point_cloud()``` you can choose between two visualization frameworks: pptk or Mayavi. 
Also, you can display the ground classification of points. Simply run:

```
cd path/to/3dhd_devkit/visualization

python point_cloud_viewer.py
```







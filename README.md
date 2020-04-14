# PointPillars Pytorch Model Convert To ONNX, And Using TensorRT to Load this IR(ONNX) for Fast Speeding Inference

Welcome to PointPillars(This is origin from nuTonomy/second.pytorch ReadMe.txt).

This repo demonstrates how to reproduce the results from
[_PointPillars: Fast Encoders for Object Detection from Point Clouds_](https://arxiv.org/abs/1812.05784) (to be published at CVPR 2019) on the
[KITTI dataset](http://www.cvlibs.net/datasets/kitti/) by making the minimum required changes from the preexisting
open source codebase [SECOND](https://github.com/traveller59/second.pytorch). 

Meanwhile, This part of the code also refers to the open source k0suke-murakami (https://github.com/k0suke-murakami/train_point_pillars) this code. 

This is not an official nuTonomy codebase, but it can be used to match the published PointPillars results.

**WARNING: This code is not being actively maintained. This code can be used to reproduce the results in the first version of the paper, https://arxiv.org/abs/1812.05784v1. For an actively maintained repository that can also reproduce PointPillars results on nuScenes, we recommend using [SECOND](https://github.com/traveller59/second.pytorch). We are not the owners of the repository, but we have worked with the author and endorse his code.**

![Example Results](https://github.com/SmallMunich/nutonomy_pointpillars/blob/master/images/pointpillars_kitti_results.png)


## Getting Started

This is a fork of [SECOND for KITTI object detection](https://github.com/traveller59/second.pytorch) and the relevant
subset of the original README is reproduced here.


### Docker Environments

If you do not waste time on pointpillars envs, please pull my docker virtual environments :

```bash
docker pull smallmunich/suke_pointpillars:v1 
```

Attention: when you launch this docker envs, please run this command :

```bash 
conda activate pointpillars 
```

And Then, you can run train or evaluation or onnx model generate command line.


### Install

#### 1. Clone code

```bash
git clone https://github.com/SmallMunich/nutonomy_pointpillars.git
```

#### 2. Install Python packages

It is recommend to use the Anaconda package manager.

First, use Anaconda to configure as many packages as possible.
```bash
conda create -n pointpillars python=3.6 anaconda
source activate pointpillars
conda install shapely pybind11 protobuf scikit-image numba pillow
conda install pytorch torchvision -c pytorch
conda install google-sparsehash -c bioconda
```

Then use pip for the packages missing from Anaconda.
```bash
pip install --upgrade pip
pip install fire tensorboardX
```

Finally, install SparseConvNet. This is not required for PointPillars, but the general SECOND code base expects this
to be correctly configured. However, I suggest you install the spconv instead of SparseConvNet.
```bash
git clone git@github.com:facebookresearch/SparseConvNet.git
cd SparseConvNet/
bash build.sh
# NOTE: if bash build.sh fails, try bash develop.sh instead
```

Additionally, you may need to install Boost geometry:

```bash
sudo apt-get install libboost-all-dev
```


#### 3. Setup cuda for numba

You need to add following environment variables for numba to ~/.bashrc:

```bash
export NUMBAPRO_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
export NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice
```

#### 4. PYTHONPATH

Add nutonomy_pointpillars/ to your PYTHONPATH.

```bash 
export PYTHONPATH=$PYTHONPATH:/your_root_path/nutonomy_pointpillars/
```

### Prepare dataset

#### 1. Dataset preparation

Download KITTI dataset and create some directories first:

```plain
└── KITTI_DATASET_ROOT
       ├── training    <-- 7481 train data
       |   ├── image_2 <-- for visualization
       |   ├── calib
       |   ├── label_2
       |   ├── velodyne
       |   └── velodyne_reduced <-- empty directory
       └── testing     <-- 7580 test data
           ├── image_2 <-- for visualization
           ├── calib
           ├── velodyne
           └── velodyne_reduced <-- empty directory
```

Note: PointPillar's protos use ```KITTI_DATASET_ROOT=/data/sets/kitti_second/```.

#### 2. Create kitti infos:

```bash
python create_data.py create_kitti_info_file --data_path=KITTI_DATASET_ROOT
```

#### 3. Create reduced point cloud:

```bash
python create_data.py create_reduced_point_cloud --data_path=KITTI_DATASET_ROOT
```

#### 4. Create groundtruth-database infos:

```bash
python create_data.py create_groundtruth_database --data_path=KITTI_DATASET_ROOT
```

#### 5. Modify config file

The config file needs to be edited to point to the above datasets:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/kitti_dbinfos_train.pkl"
    ...
  }
  kitti_info_path: "/path/to/kitti_infos_train.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
...
eval_input_reader: {
  ...
  kitti_info_path: "/path/to/kitti_infos_val.pkl"
  kitti_root_path: "KITTI_DATASET_ROOT"
}
```


### Train

```bash
cd ~/second.pytorch/second
python ./pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

* If you want to train a new model, make sure "/path/to/model_dir" doesn't exist.
* If "/path/to/model_dir" does exist, training will be resumed from the last checkpoint.
* Training only supports a single GPU. 
* Training uses a batchsize=2 which should fit in memory on most standard GPUs.
* On a single 1080Ti, training xyres_16 requires approximately 20 hours for 160 epochs.


### Evaluate


```bash
cd ~/second.pytorch/second/
python pytorch/train.py evaluate --config_path= configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

* Detection result will saved in model_dir/eval_results/step_xxx.
* By default, results are stored as a result.pkl file. To save as official KITTI label format use --pickle_result=False.

### ONNX IR Generate

### pointpillars pytorch model convert to IR onnx, you should verify some code as follows:

this python file is : second/pyotrch/models/voxelnet.py

```bash
        voxel_features = self.voxel_feature_extractor(pillar_x, pillar_y, pillar_z, pillar_i,
                                                      num_points, x_sub_shaped, y_sub_shaped, mask)

        ###################################################################################
        # return voxel_features ### onnx voxel_features export
        # middle_feature_extractor for trim shape
        voxel_features = voxel_features.squeeze()
        voxel_features = voxel_features.permute(1, 0)
```  

UNCOMMENT this line: return voxel_features 

And Then, you can run convert IR command.

```bash
cd ~/second.pytorch/second/
python pytorch/train.py onnx_model_generate --config_path= configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
```

### Compare ONNX model With Pytorch Origin model predicts 

* If you want to check this convert model about pfe.onnx and rpn.onnx model, please refer to this py-file: check_onnx_valid.py 

* Now, we can compare onnx results with pytorch origin model predicts as follows : 

* the pfe.onnx and rpn.onnx predicts file is located: "second/pytorch/onnx_predict_outputs", you can see it carefully.
```bash
    eval_voxel_features.txt 
    eval_voxel_features_onnx.txt 
    eval_rpn_features.txt 
    eval_rpn_onnx_features.txt 
```

* pfe.onnx model compare with origin pfe-layer : 
![Example Results](https://github.com/SmallMunich/nutonomy_pointpillars/blob/master/images/voxel_features.jpg)

* rpn.onnx model compare with origin rpn-layer : 
![Example Results](https://github.com/SmallMunich/nutonomy_pointpillars/blob/master/images/rpn_features.jpg)

### Compare ONNX with TensorRT Fast Speed Inference 

* First you needs this environments(onnx_tensorrt envs):

```bash
      docker pull smallmunich/onnx_tensorrt:latest
```

* If you want to use pfe.onnx and rpn.onnx model for tensorrt inference, please refer to this py-file: tensorrt_onnx_infer.py 

* Now, we can compare onnx results with pytorch origin model predicts as follows : 

* the pfe.onnx and rpn.onnx predicts file is located: "second/pytorch/onnx_predict_outputs", you can see it carefully.
```bash
    pfe_rpn_onnx_outputs.txt 
    pfe_tensorrt_outputs.txt 
    rpn_onnx_outputs.txt 
    rpn_tensorrt_outputs.txt 
```

* pfe.onnx model compare with tensorrt pfe-layer : 
![Example Results](https://github.com/SmallMunich/nutonomy_pointpillars/blob/master/images/pfe_trt.jpg)

* rpn.onnx model compare with tensorrt rpn-layer : 
![Example Results](https://github.com/SmallMunich/nutonomy_pointpillars/blob/master/images/rpn_trt.jpg)

### Blog Address

* More Details will be update on my chinese blog:
* export from pytorch to onnx IR blog : https://blog.csdn.net/Small_Munich/article/details/101559424  
* onnx compare blog : https://blog.csdn.net/Small_Munich/article/details/102073540
* tensorrt compare blog : https://blog.csdn.net/Small_Munich/article/details/102489147
* wait for update & best wishes.


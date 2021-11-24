# M3D-VTON: A Monocular-to-3D Virtual Try-On Network
Official code for ICCV2021 paper "M3D-VTON: A Monocular-to-3D Virtual Try-on Network"

[Paper](https://arxiv.org/abs/2108.05126) | [Supplementary](https://figshare.com/s/eaa35bf3a6b14f783bd5) | [MPV3D Dataset](https://drive.google.com/file/d/1qcynpXZ9eSlzTV-RDCr-Yip3GcuU314h/view?usp=sharing) | [Pretrained Models](https://figshare.com/s/fad809619d2f9ac666fc)

![M3D-VTON](/assets/teaser.gif "Teaser GIF")
## Requirements
```python >= 3.8.0, pytorch == 1.6.0, torchvision == 0.7.0```

## Data Preparation

### MPV3D Dataset
After downloading the [MPV3D Dataset](https://drive.google.com/file/d/1qcynpXZ9eSlzTV-RDCr-Yip3GcuU314h/view?usp=sharing), please run the following script to preprocess the data:
```sh
python util/data_preprocessing.py --MPV3D_root path/to/MPV3D/dataset
```

### Custom Data

If you want to process your own data, some more steps are needed (the &#8594; indicates the corresponding folder where the images should be put into):

1. prepare an in-shop clothing image *C* (&#8594; `mpv3d_example/cloth`) and a frontal person image *P* (&#8594; `mpv3d_example/image`) with resolution of 320*512;

2. obtain the mask of *C* (&#8594; `mpv3d_example/cloth-mask`) by thresholding or using [remove.bg](https://www.remove.bg/);

3. obtain the human segmentation layout (&#8594; `mpv3d_example/image-parse`) by applying [2D-Human-Paring](https://github.com/fyviezhao/2D-Human-Parsing) on *P*;

4. obtain the human joints (&#8594; `mpv3d_example/pose`) by applying [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) (25 keypoints) on *P*;

5. run the data processing script `python util/data_preprocessing.py --MPV3D_root mpv3d_example` to automatically obtain the remaining inputs (pre-aligned clothing, palm mask, and image gradients);

6. now the data preparation is finished and you should be able to run inference with the steps described in the next section "Running Inference". 

## Running Inference
We provide demo inputs under the `mpv3d_example` folder, where the target clothing and the reference person are like:

![Demo inputs](/assets/demo_inputs.png)

with inputs from the `mpv3d_example` folder, the easiest way to get start is to use the [pretrained models](https://figshare.com/s/fad809619d2f9ac666fc) and sequentially run the four steps below:

### 1. Testing MTM Module
```sh
python test.py --model MTM --name MTM --dataroot mpv3d_example --datalist test_pairs --results_dir results
```

### 2. Testing DRM Module
```sh
python test.py --model DRM --name DRM --dataroot mpv3d_example --datalist test_pairs --results_dir results
```  

### 3. Testing TFM Module
```sh
python test.py --model TFM --name TFM --dataroot mpv3d_example --datalist test_pairs --results_dir results
```

### 4. Getting colored point cloud and Remeshing

(Note: since the back-side person images are unavailable, in `rgbd2pcd.py` we provide a fast face inpainting function that produces the mirrored back-side image after a fashion. One may need manually inpaint other back-side texture areas to achieve better visual quality.)

```sh
python rgbd2pcd.py
```

Now you should get the point cloud file prepared for remeshing under `results/aligned/pcd/test_pairs/*.ply`. [MeshLab](https://www.meshlab.net/) can be used to remesh the predicted point cloud, with two simple steps below:

- Normal Estimation: Open MeshLab and load the point cloud file, and then go to Filters --> Normals, Curvatures and Orientation --> Compute normals for point sets

- Possion Remeshing: Go to Filters --> Remeshing, Simplification and Reconstruction --> Surface Reconstruction: Screen Possion (set reconstruction depth = 9)

Now the final 3D try-on result should be obtained:

![Try-on Result](/assets/meshlab_snapshot.png "Try-on Result")

## Training on MPV3D Dataset

With the pre-processed MPV3D dataset, you can train the model from scratch by folllowing the three steps below:

### 1. Train MTM module

```sh
python train.py --model MTM --name MTM --dataroot path/to/MPV3D/data --datalist train_pairs --checkpoints_dir path/for/saving/model
```

then run the command below to obtain the `--warproot` (here refers to the `--results_dir`) which is necessary for the other two modules:
```sh
python test.py --model MTM --name MTM --dataroot path/to/MPV3D/data --datalist train_pairs --checkpoints_dir path/to/saved/MTMmodel --results_dir path/for/saving/MTM/results
```

### 2. Train DRM module

```sh
python train.py --model DRM --name DRM --dataroot path/to/MPV3D/data --warproot path/to/MTM/warp/cloth --datalist train_pairs --checkpoints_dir path/for/saving/model
```

### 3. Train TFM module

```sh
python train.py --model TFM --name TFM --dataroot path/to/MPV3D/data --warproot path/to/MTM/warp/cloth --datalist train_pairs --checkpoints_dir path/for/saving/model
```

(See options/base_options.py and options/train_options.py for more training options.)

## License
The use of this code and the MPV3D dataset is RESTRICTED to non-commercial research and educational purposes.

## Citation
If our code is helpful to your research, please cite:
```
@InProceedings{M3D-VTON,
    author    = {Zhao, Fuwei and Xie, Zhenyu and Kampffmeyer, Michael and Dong, Haoye and Han, Songfang and Zheng, Tianxiang and Zhang, Tao and Liang, Xiaodan},
    title     = {M3D-VTON: A Monocular-to-3D Virtual Try-On Network},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {13239-13249}
}
```


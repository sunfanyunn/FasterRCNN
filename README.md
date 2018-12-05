# Faster-RCNN / Mask-RCNN
A minimal multi-GPU implementation of Faster-RCNN / Mask-RCNN (without FPN).
Modified from https://github.com/ppwwyyxx/tensorpack/tree/master/examples/FasterRCNN.


## Dependencies
+ Python 3; TensorFlow >= 1.4.0
+ OpenCV
+ Pre-trained [ResNet model](https://goo.gl/6XjK9V) from tensorpack model zoo.
+ OAR data. It assumes the following directory structure:
```
ct_segmentation_data_fine/
  train/*.pkl
  test/*.pkl
  val/*.pkl
  processed_image/
    /*.png
  fusion/
    /*.png
```

### File Structure
This is a minimal implementation that simply contains these files:
+ `preprocess.py`: preprocess data and save png to the directory `processed_image`. If mri image is available, images will be saved to the directory `fusion`.
+ `config.py`: configuration for all
+ `oar.py`: load OAR data
+ `data.py`: prepare data for training
+ `common.py`: common data preparation utilities
+ `basemodel.py`: implement resnet
+ `model.py`: implement rpn/faster-rcnn/mask-rcnn
+ `train.py`: main training script
+ `utils/`: third-party helper functions
+ `eval.py`: evaluation utilities
+ `viz.py`: visualization utilities

## Usage
Change config in `config.py`:
1. Set `MODE_MASK` to switch Faster-RCNN or Mask-RCNN.

Train:
```
./train.py --load /path/to/ImageNet-ResNet50.npz
```
The code is only for training with 1, 2, 4 or 8 GPUs.
Otherwise, you probably need different hyperparameters for the same performance.

Predict on an image (and show output in a window):
```
./train.py --predict input.jpg --load /path/to/model
```

### Implementation Notes

refer to https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/NOTES.md

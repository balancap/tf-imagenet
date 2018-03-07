# TensorFlow ImageNet

High performance (hopefully!) training of ImageNet TensorFlow Models.

This repository is a (shameful!) fork of the official [TensorFlow benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) source.
Whereas the latter provides a fully optimized TF benchmark on the imagenet dataset (yes, TF can be competitive with other frameworks in terms of speed!), it does not provide a full environment for obtaining the best trained models and reproducing SOTA results.

Hence, this fork focuses on providing a tested and complete implementation for training TF models on ImageNet (on deep learning stations, but also AWS P3 instances). More specifically, here are the main improvements / modifications compared to the original repo
* No additional custom layer API. Use TF slim / Keras for models definition;
* Support TF weight decay API instead of uniform L2-weight decay on every variable (which can lead a large drop of the final accuracy).
* Support `moving_average_decay`, `label_smoothing` and `gradient_clipping`  to improve accuracy;
* VGG and Inception evaluation modes;
* Additional information recorded in TensorBoard.

# State-of-the art reproduction

An important aspect of this project is to be able to reproduce SOTA results reported in the literature. Having reliable baselines has become an important subject in modern Machine Learning as improvements reported in more recent articles are not necessarily due to the introduction of new architectures, but can also be induced by different hyperparameters and training setups.

## Trained models

We have trained a couple of models to reproduce (or even improve!) results reported in the litterature. We are trying to focus on CNNs which can be used in multiple practical applications (e.g. MobileNets). Feel free to suggest some models you would to see in the following list!

Note that for relatively small models, the evaluation mode (VGG or Inception cropping) can have no negligeable impact on the top-1 and top-5 accuracies.

Publication | Model Name | Top-1 (VGG / Inception) | Top-5  (VGG / Inception) |
:----:|:------------:|:-------:|:--------:|
[MobileNets v1](https://arxiv.org/pdf/1704.04861.pdf) | [mobilenet_v1_relu](https://github.com/balancap/tf-imagenet/blob/master/models/mobilenet/mobilenet_v1_relu.py) | 72.9 / 72.2 | 90.6 / 90.5 |
[MobileNets v2 - Multiplier 1.0](https://arxiv.org/pdf/1801.04381.pdf) | [mobilenet_v2_d1](https://github.com/balancap/tf-imagenet/blob/master/models/mobilenet/mobilenet_v2.py) | 72.1 / 71.4 | 90.5 / 90.1 |
[MobileNets v2 - Multiplier 1.4](https://arxiv.org/pdf/1801.04381.pdf) | [mobilenet_v2_d14](https://github.com/balancap/tf-imagenet/blob/master/models/mobilenet/mobilenet_v2.py) | 75.0 / 74.6 | 92.0 / 91.9 |

To evaluate a checkpoint, simply use the `eval.py` script as following:
```bash
DATASET_DIR=/media/datasets/datasets/imagenet/tfrecords/
python eval.py \
    --num_gpus=1 \
    --batch_size=50 \
    --data_dir=$DATASET_DIR \
    --data_name=imagenet \
    --data_subset=validation \
    --train_dir=./checkpoints/mobilenets/mobilenets_v1_relu.ckpt \
    --ckpt_scope=v/cg/:v0/cg/ \
    --eval_method=inception \
    --data_format=NHWC \
    --moving_average_decay=0.9999 \
    --model=mobilenet_v1_relu
```

# Install

## Dependencies

* Git LFS (to get checkpoints)
* TensorFlow

## Prepare ImageNet

Download the training and evaluation archives to some `DATA_DIR`. Then, to convert to TFRecords files, simply used:
```bash
DATA_DIR=$HOME/imagenet-data
bazel build download_and_convert_imagenet
bazel-bin/download_and_convert_imagenet "${DATA_DIR}"
```

# Training

Please refer to the documentation of every model for the details on training.
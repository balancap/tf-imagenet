# TensorFlow ImageNet

High performance (hopefully!) training of ImageNet TensorFlow Models.

This repository is a (shameful!) fork of the official [TensorFlow benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) source.
Whereas the latter provides a fully optimized TF benchmark on the imagenet dataset (yes, TF can be competitive with other frameworks in terms of speed!), it does not provide a full environment for obtaining the best trained models and reproducing SOTA results.

Hence, this fork focuses on providing a tested and complete implementation for training TF models on ImageNet (on deep learning stations, but also AWS P3 instances). More specifically, here are the main improvements / modifications compared to the original repo
* No other custom layers API. Use TF slim / Keras for models definition;
* Support TF weight decay API instead of uniform L2-weight decay on every variable (which can lead a large drop of the final accuracy).
* Support `moving_average_decay`, `label_smoothing` and `gradient_clipping`  to improve accuracy;
* Additional information recorded in TensorBoard.

# State-of-the art reproduction

An important aspect of this project is to be able to reproduce SOTA results reported in the literature. Having reliable baselines has become an important subject in modern Machine Learning as improvements reported in more recent articles are not necessarily due to the introduction of new architectures, but can also be induced by different hyperparameters and training setups.

## Trained models




To evaluate a checkpoint, simply use the `eval.py` script as following:
```bash
DATASET_DIR=/media/datasets/datasets/imagenet/tfrecords/
python eval.py \
    --num_gpus=1 \
    --batch_size=50 \
    --data_dir=$DATASET_DIR \
    --data_name=imagenet \
    --data_subset=validation \
    --train_dir=./checkpoints/mobilenets_v1_relu.ckpt \
    --ckpt_scope=v/cg/:v0/cg/ \
    --data_format=NHWC \
    --moving_average_decay=0.999 \
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
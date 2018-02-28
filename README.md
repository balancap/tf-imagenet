# TensorFlow ImageNet

High performance (hopefully!) training of ImageNet TensorFlow Models.

## Dependencies

* Git LFS
* TensorFlow

## Prepare ImageNet

Download the training and evaluation archives to some `DATA_DIR`. Then, to convert to TFRecords files, simply:
```bash
DATA_DIR=$HOME/imagenet-data
bazel build download_and_convert_imagenet
bazel-bin/download_and_convert_imagenet "${DATA_DIR}"
```

## Training of models

Please refer to the documentation of every model for the details on training.
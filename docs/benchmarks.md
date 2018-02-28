# Raw benchmarks

Try to keep track of some benchmarks, to spot regressions compared to official TF benchmarks numbers, and find the sweet spot in terms of configuration.

Here is the basic command for benchmarking on the synthetic dataset:
```bash
python train.py \
    --variable_update=parameter_server \
    --local_parameter_device=gpu \
    --num_gpus=1 \
    --batch_size=32 \
    --data_format=NCHW \
    --model=inception_v2
```

## Benchmarks on GTX Titan X - TF master

Model | Dataset | Batch size | images / sec  | Comments
:------:|:-------:|:----------:|:-------------:|:-----------:
resnet v1 50 | synthetic | 32 | 139 | params on gpu
resnet v2 50 | synthetic | 32 | 138 | params on gpu
inception v1 | synthetic | 32 | 297 | params on gpu
inception v2 | synthetic | 32 | 130 | params on gpu
inception v2 | synthetic | 32 | 142 | params on gpu + NHWC
mobilenet v1 | synthetic | 32 | 154 | params on gpu
mobilenet v1 | synthetic | 32 | 176 | params on gpu + NHWC
nasnet v1 small | synthetic | 32 | 69 | params on gpu
nasnet v1 small | synthetic | 32 | 80 | params on gpu + NHWC


## Benchmarks on AWS P3 - Deep Learning AMI - TF master

Note, to get better performance, compile TensorFlow (master) on AWS P3 server (the deep learning AMI does not provide any optimized version!).
```bash
bazel build  --config=opt --copt=-march="broadwell" --config=cuda tensorflow/tools/pip_package:build_pip_package --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
```
Or manually setting the options:
```bash
bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --copt=-msse4.1 --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
```

We are currently testing with the following configuration:
```bash
python train.py \
    --variable_update=parameter_server \
    --local_parameter_device=gpu \
    --num_gpus=4 \
    --batch_size=64 \
    --data_format=NCHW \
    --model=inception_v2
```

Model | Dataset | Batch size | images / sec  | Comments
:------:|:-------:|:----------:|:-------------:|:-----------:
resnet v1 50 | synthetic | 4x64 | 1466 | params on gpu
resnet v1 50 | synthetic | 4x64 | 1472 | replicated gpu + nccl
resnet v2 50 | synthetic | 4x64 | 1440 | params on gpu
resnet v2 50 | synthetic | 4x64 | 1480 | replicated gpu + nccl
inception v2 | synthetic | 4x64 | 1162 | params on gpu
inception v2 | synthetic | 4x64 | 1510 | params on gpu + NHWC
inception v2 | synthetic | 4x64 | 1580 | replicated gpu + nccl + NHWC
mobilenet v1 | synthetic | 4x64 | 1280 | params on gpu
mobilenet v1 | synthetic | 4x64 | 2248 | params on gpu + NHWC
mobilenet v1 | synthetic | 4x64 | 1264 | replicated gpu + nccl
mobilenet v1 | synthetic | 4x64 | 2177 | replicated gpu + nccl + NHWC
nasnet v1 small | synthetic | 4x64 | 591 | replicated gpu + nccl
nasnet v1 small | synthetic | 4x64 | 857 | replicated gpu + nccl + NHWC

# ImageNet training benchmarks


```bash
python train.py \
    --variable_update=replicated \
    --local_parameter_device=gpu \
    --print_training_accuracy=true \
    --gpu_memory_frac_for_testing=0.8 \
    --tf_random_seed=1234 \
    --summary_verbosity=1 \
    --resize_method=bilinear \
    --optimizer=sgd \
    --learning_rate=0.05 \
    --num_epochs_per_decay=2 \
    --learning_rate_decay_factor=0.94 \
    --weight_decay=0.00001 \
    --distortions=true \
    --num_gpus=4 \
    --batch_size=64 \
    --data_format=NCHW \
    --model=inception_v2
```


Model | Dataset | Batch size | images / sec  | Comments
:------:|:-------:|:----------:|:-------------:|:-----------:
resnet v1 50 | synthetic | 32 | 315 |
resnet v2 50 | synthetic | 32 | 313 |
inception v1 | synthetic | 32 | 608 |
inception v2 | synthetic | 32 | 298 |
mobilenet v1 | synthetic | 32 | 305 | TF master used
nasnet v1 small | synthetic | 32 | 135 |

## Benchmarks on AWS P3 - NVIDIA docker images

These morons don't even have TF 1.5 / master available!

# ImageNet accuracy

Standard command for benchmarking a model on ImageNet:
```bash
python train.py \
    --eval=True \
    --variable_update=parameter_server \
    --local_parameter_device=gpu \
    --num_gpus=1 \
    --num_intra_threads=4 \
    --num_inter_threads=0 \
    --data_name=imagenet \
    --data_dir=/media/paul/DataExt4/ImageNet/dataset \
    --train_dir=./checkpoints/inception_v2_fused.ckpt \
    --ckpt_scope=v/cg/: \
    --batch_size=50 \
    --num_batches=1000 \
    --model=inception_v2
```

Model | Top-1 | Top-5 | Pre-processing  | Comments
:------:|:-------:|:----------:|:-------------:|:-----------:
resnet v1 50 |  |  | VGG-slim |

# ResNet training

A few tips on training ResNet networks.

## Training
```bash
DATASET_DIR=/media/paul/DataExt4/ImageNet/dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/logs/resnet_v2_50_002
python train.py \
    --variable_update=parameter_server \
    --local_parameter_device=gpu \
    --print_training_accuracy=true \
    --gpu_memory_frac_for_testing=0.8 \
    --tf_random_seed=1234 \
    --summary_verbosity=3 \
    --save_summaries_steps=1000 \
    --save_model_secs=1800 \
    --num_intra_threads=4 \
    --num_inter_threads=0 \
    --data_format=NCHW \
    --data_name=imagenet \
    --data_dir=${DATASET_DIR} \
    --train_dir=${TRAIN_DIR} \
    --display_every=10 \
    --num_batches=250000 \
    --batch_size=32 \
    --num_gpus=1 \
    --resize_method=bilinear \
    --optimizer=sgd \
    --learning_rate=0.1 \
    --num_epochs_per_decay=2 \
    --learning_rate_decay_factor=0.94 \
    --weight_decay=0.00004 \
    --distortions=true \
    --model=resnet_v2_50
```

## Evaluation

```bash
DATASET_DIR=/media/paul/DataExt4/ImageNet/dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/logs/resnet_v2_50_002
python train.py \
    --eval=True \
    --variable_update=parameter_server \
    --local_parameter_device=gpu \
    --print_training_accuracy=true \
    --gpu_memory_frac_for_testing=0.1 \
    --tf_random_seed=1234 \
    --summary_verbosity=1 \
    --num_gpus=1 \
    --num_intra_threads=2 \
    --num_inter_threads=0 \
    --data_name=imagenet \
    --data_dir=${DATASET_DIR} \
    --train_dir=${TRAIN_DIR} \
    --eval_dir=${TRAIN_DIR}/eval \
    --batch_size=10 \
    --num_batches=100 \
    --model=resnet_v2_50
```
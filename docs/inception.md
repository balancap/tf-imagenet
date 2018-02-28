# Inception V2 training

## Training

```bash
DATASET_DIR=/media/datasets/datasets/imagenet/tfrecords/
TRAIN_DIR=/home/ubuntu/logs/inception_v2/0001
python train.py \
    --variable_update=replicated \
    --all_reduce_spec=nccl \
    --local_parameter_device=gpu \
    --print_training_accuracy=true \
    --gpu_memory_frac_for_testing=0.8 \
    --tf_random_seed=1234 \
    --summary_verbosity=1 \
    --save_summaries_steps=200 \
    --save_model_secs=1800 \
    --num_intra_threads=20 \
    --num_inter_threads=64 \
    --data_format=NCHW \
    --data_name=imagenet \
    --data_dir=${DATASET_DIR} \
    --train_dir=${TRAIN_DIR} \
    --display_every=10 \
    --num_batches=1000000 \
    --batch_size=64 \
    --num_gpus=4 \
    --resize_method=bilinear \
    --optimizer=sgd \
    --learning_rate=0.05 \
    --num_epochs_per_decay=2 \
    --learning_rate_decay_factor=0.94 \
    --weight_decay=0.00001 \
    --distortions=true \
    --model=inception_v2
```

```bash
DATASET_DIR=/media/paul/DataExt4/ImageNet/dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/logs/resnet_v2_50_002
python train.py \
    --variable_update=replicated \
    --print_training_accuracy=true \
    --gpu_memory_frac_for_testing=0.8 \
    --tf_random_seed=1234 \
    --summary_verbosity=1 \
    --save_summaries_steps=200 \
    --save_model_secs=1800 \
    --num_intra_threads=24 \
    --num_inter_threads=64 \
    --data_format=NCHW \
    --data_name=imagenet \
    --data_dir=${DATASET_DIR} \
    --train_dir=${TRAIN_DIR} \
    --display_every=50 \
    --num_batches=250000 \
    --batch_size=64 \
    --num_gpus=4 \
    --resize_method=bilinear \
    --optimizer=rmsprop \
    --learning_rate=0.05 \
    --num_epochs_per_decay=2 \
    --learning_rate_decay_factor=0.94 \
    --gradient_clip=None \
    --weight_decay=0.00002 \
    --distortions=true \
    --model=inception_v2
```

## Evaluation

```bash
DATASET_DIR=/media/paul/DataExt4/ImageNet/dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/logs/resnet_v2_50_002
python eval.py --train_dir=${TRAIN_DIR}  --ckpt_scope=v/cg/:v0/cg/
```

# Inception V3 training

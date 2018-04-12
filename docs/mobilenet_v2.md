# MobileNet v2 training

```bash
python train.py \
    --variable_update=replicated \
    --all_reduce_spec=nccl \
    --local_parameter_device=cpu \
    --print_training_accuracy=true \
    --gpu_memory_frac_for_testing=0.8 \
    --tf_random_seed=1234 \
    --summary_verbosity=1 \
    --save_summaries_steps=200 \
    --save_model_secs=1800 \
    --num_intra_threads=0 \
    --num_inter_threads=0 \
    --data_format=NHWC \
    --data_name=imagenet \
    --data_dir=${DATASET_DIR} \
    --train_dir=${TRAIN_DIR} \
    --display_every=10 \
    --num_batches=10000000 \
    --batch_size=64 \
    --num_gpus=4 \
    --resize_method=bilinear \
    --optimizer=rmsprop \
    --learning_rate=0.2 \
    --num_epochs_per_decay=2 \
    --learning_rate_decay_factor=0.97 \
    --weight_decay=0.00004 \
    --moving_average_decay=0.9999 \
    --label_smoothing=0.1 \
    --gradient_clip=10.0 \
    --distortions=true \
    --model=mobilenet_v2_d1
```

```bash
nohup python eval.py --train_dir=${TRAIN_DIR}  --ckpt_scope=v/cg/:v0/cg/  --moving_average_decay=0.9999 &
```

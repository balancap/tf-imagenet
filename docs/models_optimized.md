# Optimization of ImageNet models

## Inception 2

Here are some training results.

Model | Batch size | Epochs | LR + params | Dilations | Shuffling | Top-1 | Top-5 | Remarks
:----:|:----------:|:------:|:-----------:|:---------:|:---------:|:-----:|:-----:|:-----:|
Inception v2b | 256 | 280 | rmpsprop 0.2 + 0.97 + 0.00004 | [1, [2, 1]] | No | 73.0 | 91.1 | Aux Logits + label smoothing + grad clipping at 10
Inception v2b | 256 | 430 | rmpsprop 0.3 + 0.97 + 0.00004 | [1, [3, 2]] | No | 73.0 | 91.0 | Aux Logits + Label smoothing + grad clipping at 10
Inception v2b | 256 | 320 | rmpsprop 0.3 + 0.97 + 0.00004 | [1, [4, 1]] + [1, [2, 1]] | No | 72.5 | 90.9 | Aux Logits + label smoothing + grad clipping at 10

Inception v2b | 256 | - | rmpsprop 0.2 + 0.97 + 0.00004 | [1, [2, 2]] | No | - | - | Aux Logits + label smoothing + grad clipping at 10
Inception v2b | 256 | - | rmpsprop 0.2 + 0.97 + 0.00004 | [1, [2, 1]] + [1, [4, 2]] | No | - | - | Aux Logits + label smoothing + grad clipping at 10
Inception v2b | 256 | - | rmpsprop 0.2 + 0.97 + 0.00004 | [1, [2, 4]] | No | - | - | Aux Logits + label smoothing + grad clipping at 10

A few general observations:
* large dilations do not seem to help much, on contrary;
* NASNet training setup has worse performance at the beginning, but pretty good one at the end;


## Nasnet
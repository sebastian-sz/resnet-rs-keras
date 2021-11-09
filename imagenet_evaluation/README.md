# Imagenet Evaluation
Below you can find results for imagenet evaluation, obtained by running `main.py` in 
this part of repository.

### Results

#### Notice
Please bear in mind that the models in this repository, produce the same outputs as the
models exported from `tensorflow/tpu` repository (assuming the same input and machine).

The Imagenet Validation Accuracy is expected to fluctuate a bit when
rewriting the models between API's or frameworks.

#### Table

| Depth | Image size | Official Top 1 | This repo Top 1 | This repo Top 5 | Top 1 diff |
| ----- | ---------- | -------------- | --------------- | --------------- | ---------- |
|  50   |    160     |      78.8      |        78.9     |       94.3      |    +0.1    |
|  101  |    160     |      80.3      |        80.3     |       95.0      |     0.0    |
|  101  |    192     |      81.2      |        81.3     |       95.6      |    +0.1    |
|  152  |    192     |      82.0      |        82.1     |       95.9      |    +0.1    |
|  152  |    224     |      82.2      |        82.9     |       96.2      |    +0.7    |
|  152  |    256     |      83.0      |        83.0     |       96.3      |     0.0    |
|  200  |    256     |      83.4      |        83.4     |       96.5      |     0.0    |
|  270  |    256     |      83.8      |        83.8     |       96.7      |     0.0    |
|  350  |    256     |      84.0      |        84.0     |       96.8      |     0.0    |
|  350  |    320     |      84.2      |        84.3     |       96.8      |    +0.1    |
|  420  |    320     |      84.4      |        84.4     |       96.8      |     0.0    |


#### Why the difference
One can speculate. Although not large, the difference might come from:
* The API used: Official uses `TPUEstimator`, I use `tf.keras.Model`
* Hardware used: Official uses TPU, I use GPU.
* Precision: Official runs in `bfloat16`, I use `float32`.

### To reproduce my eval:
The `external/` directory contains code from original repository.

1. Download Imagenet and use [imagenet_to_gcs](https://github.com/tensorflow/tpu/blob/acb331c8878ce5a4124d4d7687df5fe0fadcd43b/tools/datasets/imagenet_to_gcs.py) 
script to obtain tfrecords:
```
python imagenet_to_gcs.py \
    --raw_data_dir=imagenet_home/ \
    --local_scratch_dir=my_tfrecords \
    --nogcs_upload
```
        
2. To eval this repo models, run  (for example):
```
python main.py  \
    --depth 50 \
    --data_dir /path/to/tfrecords/validation \
    --weights /path/to/weights \  # Or weight argument
    --image_size 224
```   

Change parameters accordingly, as in the table above.

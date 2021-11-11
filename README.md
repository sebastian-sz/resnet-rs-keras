Resnet-RS models rewritten in Tensorflow / Keras functional API.

# Table of contents
1. [Introduction](https://github.com/sebastian-sz/resnet-rs-keras#introduction)
2. [Quickstart](https://github.com/sebastian-sz/resnet-rs-keras#quickstart)
3. [Installation](https://github.com/sebastian-sz/resnet-rs-keras#installation)
4. [How to use](https://github.com/sebastian-sz/resnet-rs-keras#how-to-use)
5. [Original Weights](https://github.com/sebastian-sz/resnet-rs-keras#original-weights)

# Introduction
This is a package with ResNet-RS models adapted to Tensorflow/Keras.  

ResNet-RS models are updated versions of ResNet models - [Arxiv Link](https://arxiv.org/abs/2103.07579)  
The model's weights are converted from [original repository](https://github.com/tensorflow/tpu/blob/acb331c8878ce5a4124d4d7687df5fe0fadcd43b/models/official/resnet/resnet_rs/).

# Quickstart
The design was meant to mimic the usage of `keras.applications`:
```python
!pip install git+https://github.com/sebastian-sz/resnet-rs-keras@main

# Import package:
from resnet_rs import ResNetRS50
import tensorflow as tf

# Use model directly:
model = ResNetRS50(weights='imagenet', input_shape=(224, 224, 3))
model.summary()

# Or to extract features / fine tune:
backbone = ResNetRS50(
   weights='imagenet', 
   input_shape=(224,224, 3),
   include_top=False
)
model = tf.keras.Sequential([
    backbone,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)  # 10 = num classes
])
model.compile(...)
model.fit(...)
```
You can fine tune these models, just like other Keras models.  

For end-to-end fine-tuning and conversion examples check out the 
[Colab Notebook](https://colab.research.google.com/drive/1FQAGBj4RwkjZcPckYkO0YAcrCeKYN26Z?usp=sharing).

# Installation
There are multiple ways to install.  
The only requirements are Tensorflow 2.4+ and Python 3.6+.  

### Option A: (recommended) pip install from GitHub
`pip install git+https://github.com/sebastian-sz/resnet-rs-keras@main`

### Option B: Build from source
```bash
git clone https://github.com/sebastian-sz/resnet-rs-keras.git  
cd resnet-rs-keras  
pip install .
```

### Option C: (alternatively) no install:
If you do not want to install you could just drop the `resnet_rs/` directory directly into your project.

### Option D: Docker
You can also install this package as an extension to official Tensorflow docker container:  

Build: `docker build -t resnet_rs_keras .`  
Run: `docker run -it --rm resnet_rs_keras`

For GPU support or different TAG you can (for example) pass  
`--build-arg IMAGE_TAG=2.5.0-gpu`  
in build command.

### Verify installation
If all goes well you should be able to import:  
`from resnet_rs import *`

# How to use
There are 7 model variants you can use, with the following depths:   
`ResNetRS[50, 101, 152, 200, 270, 350, 420]` 

### Imagenet weights
The imagenet weights are automatically downloaded if you pass `weights="imagenet"` 
option while creating the models.   

Note: for a single depth, sometimes multiple weight variants have been released, 
depending on the input shape the network has been trained with. By **default**
the highest input shape weights are downloaded as they yield the best accuracy.

### Possible weight variants:
As of writing this repository the following weight variants are available:
```
50: [160],
101: [160, 192],
152: [192, 224, 256],
200: [256],
270: [256],
350: [256, 320],
420: [320],
```
To pick a specific weight variant (instead of default / highest) pass, `imagenet-i<shape>` as a weight argument.
For example, to create ResNetRS152, with weights trained with 224 input shape, run:
`model = ResNetRS152(weights="imagenet-i224")`

### Input shapes
**Weights are input shape agnostic**. You can create a model with different input 
shape than what it was trained with:   
```python
# Trained with 320x320 
# Now has 224x224
model = ResNetRS350(weights="imagenet-i320", input_shape=(224, 224, 3))
```  
By default, it is suggested to use shape 224x224.
### Preprocessing
The models expect image, normalized with Imagenet's mean and stddev:  
```python
import tensorflow as tf

mean_rgb = [0.485, 0.456, 0.406]
stddev_rgb = [0.229, 0.224, 0.225]

def preprocess(image):  # input image is in range 0-255.
    image = image / 255.
    image -= tf.constant(mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
    image /= tf.constant(stddev_rgb, shape=[1, 1, 3], dtype=image.dtype)
    return image
```

##### (Alternatively) Preprocessing Layer:
Or you can use [Preprocessing Layer](https://keras.io/guides/preprocessing_layers/):
```python
from resnet_rs import get_preprocessing_layer

layer = get_preprocessing_layer()
inputs = layer(image)
```

### Fine-tuning
For fine-tuning example, check out the [Colab Notebook](https://colab.research.google.com/drive/1FQAGBj4RwkjZcPckYkO0YAcrCeKYN26Z?usp=sharing).

### Tensorflow Lite
The models are TFLite compatible. You can convert them like any other Keras model:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("resnet_rs.tflite", "wb") as file:
  file.write(tflite_model)
```

### ONNX
The models are ONNX compatible. For ONNX Conversion you can use [tf2onnx](
https://github.com/onnx/tensorflow-onnx) package:
```python
!pip install tf2onnx==1.8.4

# Save the model in TF's Saved Model format:
model.save("my_saved_model/")

# Convert:
!python -m tf2onnx.convert \
  --saved-model my_saved_model/ \
  --output resnet_rs.onnx
```

# Original Weights
The original weights are present in the [original repository](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs)
in the form of `ckpt` files. I converted the weights to Keras `.h5` format so it's 
easier to use with Tensorflow/Keras API.

### (Optionally) Convert the weights
The converted weights are on this repository's GitHub. If, for some reason, you wish to 
download and convert original weights yourself, I prepared the utility scripts: 
1. `bash scripts/download_all.sh`
2. `bash scripts/convert_all.sh`
   
# Closing words
If you found this repo useful, please consider giving it a star!

# Bibliography
1. [Original Repository](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs)
2. [Keras Reference Implementation](https://github.com/tensorflow/models/blob/master/official/vision/beta/modeling/backbones/resnet.py)
# CV_Second_Assignment
Second assignment of Computer vision course for emotion recognition

## Project setup
```
git clone https://github.com/Aenteas/CV_Second_Assignment
```

```
cd project root
```

```
pip3 install -r requirements.txt
```

download fer2013 dataset

```
mkdir data
```

Place the fer2013.csv file inside data folder

## Usage

### Train model

```
python3 run.py -m [model_name]
```

There are 5 different models:
[model_name] = vgg2_0_2, vgg2_2_2, vgg2_4_ilrb_2, vgg2_4_2_conv3, vgg2_4_2_conv5

For details read models.py

Early stopping is used for validation (--patience and --num_epoch_to_validate arguments)

Batch size, learning rate and other parameters can be specified for training (see run.py)
Trained models are saved under checkpoint folder
Training can be run on checkpoint models using the --checkpoint argument
After training, inference is run on the test set, computing test loss and accuracy, showing the confusion matrix and saving predictions under outputs folder
The -wl flag adds weighted cross entropy loss to mitigate imbalance in the dataset

### Run inference

```
python3 infer.py --checkpoint [path_to_checkpoint model]
```

Computes the confusion matrix on test set and saving results under outputs folder

## Models

There are 5 different models to train, each uses the first 2 convolutional blocks of a pretrained VGG11 model. The clasification block in each model consists of 2 layers with 512 hidden units and 7 output units (number of emotions). The intermediate layers of individual models are as follows:

* vgg2_0_2: A single adaptive average pooling reducing the spatial size to 6 by 6 before the classification module.

* vgg2_2_2: 2 convolutional layers of 3 by 3 kernel size, with pooling operations in between.

* vgg2_4_ilrb_2: 4 inverted residual blocks with linear bottlenecks adopted from MobileNETv2 with pooling operations. Inverted residual blocks are realized by a layer of 1 by 1 convolution increasing the number of channels (expansion) followed by 3 by 3 depthwise convolution reducing the depth of the output tensor. The authors added linear bottlenecks to prevent non-linearities  from  destroying  too  much  information and skip connection between the input and output in case the the input and output tensor match in size.

* vgg2_4_2_conv3: 4 convolutional layers of 3 by 3 kernel size, with pooling operations.

* vgg4_2_2_conv5: 4 convolutional layers of 5 by 5 kernel size, with a single max pooling operation.
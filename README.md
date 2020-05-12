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
[model_name] = vgg4_0_2, vgg4_2_2, vgg4_2_ilrb_2, vgg4_2_2_2maxpool, vgg4_2_2_conv5

For details read models.py

Early stopping is used for validation (--patience and --num_epoch_to_validate arguments)

Batch size, learning rate and other parameters can be specified for training (see run.py)
Trained models are saved under checkpoint folder
Training can be run on checkpoint models using the --checkpoint argument
After training, inference is run on the test set, showing the confusion matrix and saving results under outputs folder

### Run inference

```
python3 infer.py --checkpoint [path_to_checkpoint model]
```

Computes the confusion matrix on test set and saving results under outputs folder

## Models

There are 5 different models to train, each uses the first 4 convolutional layers of a pretrained VGG11 model. The clasification block in each model consists of 2 layers with 512 hidden units and 7 output units (number of emotions). 

* vgg4_0_2: First 4 convolutional layers of VGG11, average pooling and classification block

* vgg4_2_2: First 4 convolutional layers of VGG11, 3x3 convolutional layer, max pooling, 3x3 convolutional layer and classification block

* vgg4_2_ilrb_2: First 4 convolutional layers of VGG11, 2 inverted linear residual blocks separated by max pooling and classification block

* vgg4_2_2_2maxpool: First 4 convolutional layers of VGG11, 3x3 convolutional layer, max pooling, 3x3 convolutional layer, max pooling and classification block

* vgg4_2_2_conv5: First 4 convolutional layers of VGG11, 2 5x5 convolutional layers and classification block
# GANja
## Overview
A GAN to create art

![](https://github.com/mikepatel/GANja/blob/main/overview.png)

## Data
* temporarily using [shoe images](https://github.com/mikepatel/GANja/tree/main/data/train), but this will change

## Environment
* Python 3.7
* Anaconda environment
  * TensorFlow 2.1

## File descriptions
* [model.py](https://github.com/mikepatel/GANja/blob/main/model.py) - for model definitions
* [parameters.py](https://github.com/mikepatel/GANja/blob/main/parameters.py) - for model and training parameters
* [train.py](https://github.com/mikepatel/GANja/blob/main/train.py) - for preprocessing and training

## Instructions
### To train model
```
$ python train.py
```

## Additional notes
* currently not saving a trained model as training is still ongoing
* will use a different dataset in the future, possibly from [WikiArt](https://www.wikiart.org/)

### Some things to try
* make custom discriminator and generator models smaller (less deep)
* use VGG16 classifier instead of a custom discriminator model
* use lower resolution images as the input to the generator, and upsample more over training
* use a ProGAN approach to model design

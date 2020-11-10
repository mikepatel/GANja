# GANja
## Overview
Use ML to create digital art

### Authors
* [John Jefferson III](https://github.com/jjefferson1994)
* Daniel Rodriguez
* [Michael Patel](https://github.com/mikepatel)

## Data
* Using images from Wikimedia commons
* Sample images in [style transfer/data](https://github.com/mikepatel/GANja/tree/main/style%20transfer/data)

## Environment
* Python 3.7
* Anaconda environment
  * TensorFlow 2.1

## File descriptions
* [style transfer/run.py](https://github.com/mikepatel/GANja/blob/main/style%20transfer/run.py)

## Instructions
### To train model
TO DO

### To run model and generate output
```
$ python style transfer/run.py
```

## Preliminary results
| Content image input | Style image input | Generated output image |
:------------:|:------------:|:------------:
![content](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/content/content.jpg) | ![style](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/style/style.jpg) | ![output](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/generated/generated_06-11-2020_16-41-32.jpg)

## Additional notes
* currently not saving a trained model as training is still ongoing
* will use a different dataset in the future, possibly from [WikiArt](https://www.wikiart.org/)

### Some things to try
* make custom discriminator and generator models smaller (less deep)
* use VGG16 classifier instead of a custom discriminator model
* use lower resolution images as the input to the generator, and upsample more over training
* use a ProGAN approach to model design

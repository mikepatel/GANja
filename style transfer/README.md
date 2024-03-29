# GANja - style transfer
## Overview
Use ML style transfer to create digital art

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

## Files
* [style transfer/run.py](https://github.com/mikepatel/GANja/blob/main/style%20transfer/run.py)

## Instructions
### To run model and generate output using style transfer technique
1. Load content images into directory [data/content](https://github.com/mikepatel/GANja/tree/main/style%20transfer/data/content)
2. Load style images into directory [data/style](https://github.com/mikepatel/GANja/tree/main/style%20transfer/data/style)
3. Run python script [style transfer/run.py](https://github.com/mikepatel/GANja/blob/main/style%20transfer/run.py)
```
$ python "style transfer/run.py"
```
4. Generated images (.jpg) are saved into directory [data/generated](https://github.com/mikepatel/GANja/tree/main/style%20transfer/data/generated)

## Preliminary results
| Content image input | Style image input | Generated output image |
:------------:|:------------:|:------------:
![content](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/content/content_hill.jpg) | ![style](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/style/style_wave.jpg) | ![output](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/generated/generated_hill_wave.jpg)
![content](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/content/content_dog.jpg) | ![style](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/style/style_wave.jpg) | ![output](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/generated/generated_dog_wave.jpg)
![content](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/content/content_ship.jpg) | ![style](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/style/style_wave.jpg) | ![output](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/generated/generated_ship_wave.jpg)
![content](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/content/content_hill.jpg) | ![style](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/style/pastel.jpg) | ![output](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/generated/generated_hill_pastel.jpg)
![content](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/content/content_dog.jpg) | ![style](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/style/pastel.jpg) | ![output](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/generated/generated_dog_pastel.jpg)
![content](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/content/content_ship.jpg) | ![style](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/style/pastel.jpg) | ![output](https://github.com/mikepatel/GANja/blob/main/style%20transfer/data/generated/generated_ship_pastel.jpg)

## Additional notes
* use a different dataset in the future, possibly from [WikiArt](https://www.wikiart.org/)

### Some things to try
* use higher resolution style images
* Try a GAN
  * make custom discriminator and generator models smaller (less deep)
  * use VGG16 classifier instead of a custom discriminator model
  * use lower resolution images as the input to the generator, and upsample more over training
  * use a ProGAN approach to model design

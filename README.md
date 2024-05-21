# Face alignment for PCR students

#### Requisites
- images_framework https://github.com/pcr-upm/images_framework
- scipy
- tensorflow[and-cuda]
- keras
- nvidia-tensorrt

#### Installation
This repository must be located inside the following directory:
```
images_framework
    └── alignment
        └── pami20_reconstruction
```
#### Usage
```
usage: pami20_reconstruction_test.py [-h] [--input-data INPUT_DATA] [--show-viewer] [--save-image]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --show-viewer option to show results visually.

* Use the --save-image option to save the processed images.
```
usage: Alignment --database DATABASE
```

* Use the --database option to select the database model.
```
usage: Pami20Reconstruction [--gpu GPU]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).
```
> python images_framework/alignment/pami20_reconstruction/test/pami20_reconstruction_test.py --input-data images_framework/alignment/pami20_reconstruction/test/example.tif --database aflw --gpu 0 --save-image
```
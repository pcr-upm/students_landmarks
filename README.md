# Face alignment for PCR students

#### Requisites
- images_framework https://github.com/pcr-upm/images_framework
- scipy
- torch
- pytorch-lightning
- torchvision
- torch-summary
- tensorboard

#### Installation
This repository must be located inside the following directory:
```
images_framework
    └── alignment
        └── students_landmarks
```
#### Usage
```
usage: students_landmarks_test.py [-h] [--input-data INPUT_DATA] [--show-viewer] [--save-image]
```

* Use the --input-data option to set an image, directory, camera or video file as input.

* Use the --show-viewer option to show results visually.

* Use the --save-image option to save the processed images.
```
usage: Alignment --database DATABASE
```

* Use the --database option to select the database model.
```
usage: StudentsLandmarks [--gpu GPU] --backbone {SHG} [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--patience PATIENCE]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).

* Use the --backbone option to set the backbone model.

* Use the --batch-size option to set the number of images in each mini-batch.

* Use the --epochs option to set the number of sweeps over the dataset to train.

* Use the --patience option to set number of epochs with no improvement after which training will be stopped.
```
> python images_framework/alignment/students_landmarks/test/students_landmarks_test.py --input-data images_framework/alignment/students_landmarks/test/example.tif --database wflw --gpu 0 --backbone shg --save-image
```
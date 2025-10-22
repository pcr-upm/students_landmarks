# Face alignment for PCR students

#### Requisites
- images-framework
- torch
- pytorch-lightning
- torchvision
- torch-summary
- tensorboard
- segmentation-models-pytorch
- tqdm
- scikit-learn

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
usage: StudentsLandmarks [--gpu GPU] --regressor {encoder,unet} --backbone {resnet18,resnet34,resnet50,resnet101,resnet152,efficientnet-b0,efficientnet-b1,efficientnet-b2,efficientnet-b3,efficientnet-b4,efficientnet-b5,efficientnet-b6,efficientnet-b7,vit} [--batch-size BATCH_SIZE] [--epochs EPOCHS] [--patience PATIENCE]
```

* Use the --gpu option to set the GPU identifier (negative value indicates CPU mode).

* Use the --regressor option to set the regressor model.

* Use the --backbone option to set the backbone architecture.

* Use the --batch-size option to set the number of images in each mini-batch.

* Use the --epochs option to set the number of sweeps over the dataset to train.

* Use the --patience option to set number of epochs with no improvement after which training will be stopped.
```
> python test/students_landmarks_train.py --anns-file wflw_ann_train.txt --database wflw --gpu 0 --regressor encoder --backbone resnet50 --batch-size 64 --epochs 100 --patience 20
```
```
> python test/students_landmarks_test.py --input-data test/example.tif --database wflw --gpu 0 --regressor encoder --backbone resnet50 --save-image
```
```
> python test/students_landmarks_database.py --anns-file wflw_ann_test.txt --database wflw --gpu 0 --regressor encoder --backbone resnet50 --save-file
```
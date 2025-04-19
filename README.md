# inatrualist-conv-nets
Training and finetuning convolutional networks on a subset of inaturalist dataset.


## Base module and data loaders

The file `trainer.py` contains the data loader logic for the subset of inaturalist dataset from the following [link](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) and the base training, validation and testing hooks as a pytorch lightning module.


# Training convolutional network form scratch

A simple convolutional neural network is defined as lightning module called `ConvNN` in train.py which extends the `BaseModule` from the `trainer.py`. To train the convolutional network train py can be run with the required arguments.

# Finetuning ResNet50

The `ResnetFinetune` module which extends the `BaseModule` implements a finetuner for resnet50 model which is implemented in `resnet_finetune.py`

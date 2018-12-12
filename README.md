# Neural scene representation and rendering (Eslami, et al., 2018)
![img](https://storage.googleapis.com/deepmind-live-cms/images/model.width-1100.png)

## Requirement
- Python >=3.6
- Pytorch
- TensorBoardX

## How to train
```
python train.py --train_data_dir /path/to/dataset/train --test_data_dir /path/to/dataset/test

# if you use multiple GPUs
python train.py --device_ids 0 1 2 3 --train_data_dir /path/to/dataset/train --test_data_dir /path/to/dataset/test
```

## Usage
### representation.py
Representation networks (See Figure S1 in Supplementary Materials of the paper).

### core.py
Core networks of inference and generation (See Figure S2 in Supplementary Materials of the paper).

### conv_lstm.py
Implementation of convolutional LSTM used in `core.py`.

### gqn_dataset.py
Dataset class.

### model.py
Main module of Generative Query Network.

### train.py
Training algorithm.

### scheduler.py
Scheduler of learning rate used in `train.py`.

## Results (WIP)
|Ground Truth|Generation|
|![ground_truth](https://user-images.githubusercontent.com/24241353/49865725-100aa180-fe49-11e8-9ae4-cd9ed54a6bc2.png)|



## Setup
### Install Package Dependencies
```
torch
torchvision 
PyYAML
easydict
tqdm
scikit-learn
efficientnet_pytorch
pandas
opencv
```
### Datasets
***create a soft link to the dataset directory***

CUB dataset
```
ln -s /your-path-to/CUB-dataset data/cub
```



## Training (Backbone: ResNet50, Dataset: CUB)

``` python main.py --config config/sps-single-branch-cub.yml --gpu_ids 0 ``` # Train the network that contains one mid-level branch 

Best accuracy: 88.70%

Results of the last three epochs

|  Epoch | H-level | M-level(SPS) | H-level+M-level(SPS) |  
|:--------|:--------|--------:|------:|
|158|85.76%|87.21%|88.51%|
|159|86.18%|87.37%|88.32%|
|160|85.99%|87.33%|88.63%|



``` python main.py --config config/sps-two-branch-cub.yml --gpu_ids 0 ``` # Train the network that contains two mid-level branches

Best accuracy: 88.82%

Results of the last three epochs

|  Epoch | H-level | M-level(SPS)-0| M-level(SPS)-1 | H-level+2xM-level(SPS) |  
|:--------|:--------|--------:|------:|------:|
|158|85.76%|87.82%|87.33%|88.64%|
|159|85.83%|87.66%|87.28%|88.57%|
|160|85.42%|87.68%|88.52%|88.70%|

*** The full training log can be found in the folder logs/ ***




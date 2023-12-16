# PyTorch implementation of "RankMixup: Ranking-Based Mixup Training for Network Calibration"

This is the implementation of the paper "RankMixup: Ranking-Based Mixup Training for Network Calibration (ICCV 2023)".

For more information, checkout our project site [[website](https://cvlab.yonsei.ac.kr/projects/RankMixup/)] or our paper [[PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Noh_RankMixup_Ranking-Based_Mixup_Training_for_Network_Calibration_ICCV_2023_paper.pdf)].

## Dependencies
* Python >= 3.8
* PyTorch == 1.8.1

## Datasets
For CIFAR-10, the dataset will be automatically downloaded when you run the code. For the others (Tiny-Imagenet, CUB-200 and VOC 2012), please download the datasets using the official cites.
And please add the absolute path of the data directory for the corresponding data configs located in configs/data. 

## Installation
```python
pip install -e .
```

## Training
The config files for our loss functions are in configs/loss/mrl.yaml or mndcg.yaml.
<details><summary>Tiny-ImageNet</summary>
<p>

You can easily train your own model like:
  ```bash
  sh train_tiny.sh 
  ```
or You can freely define parameters with your own settings like:
  ```python
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    log_period=100 \
    data=tiny_imagenet \
    data.data_root='your_dataset_directory' \
    model=resnet50_mixup_tiny model.num_classes=200 \
    loss=mrl or mndcg \
    optim=sgd optim.lr=0.1 optim.momentum=0.9 \
    scheduler=multi_step scheduler.milestones="[40, 60]" \
    train.max_epoch=100 
  ```
</p>
</details>


<details><summary>CIFAR10</summary>
<p>

You can easily train your own model like:
  ```bash
  sh train_cifar.sh 
  ```
or You can freely define parameters with your own settings like:
  ```python
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
    log_period=100 \
    data=cifar10 \
    data.data_root='your_dataset_directory' \
    model=resnet50_mixup model.num_classes=10 \
    loss=mrl or mndcg \
    optim=sgd optim.lr=0.1 optim.momentum=0.9 \
    scheduler=multi_step scheduler.milestones="[80, 120]" \
    train.max_epoch=200 
  ```

</p>
</details>

## Testing
<details><summary>Tiny-ImageNet</summary>
<p>

```python
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    log_period=100 \
    data=tiny_imagenet \
    data.data_root='your_dataset_directory' \
    model=resnet50_mixup_tiny model.num_classes=200 \
    loss=mrl or mndcg \
    hydra.run.dir='your_bestmodel_directory' \
    test.checkpoint=best.pth \
```

</p>
</details>

<details><summary>CIFAR10</summary>
<p>

```python
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
    log_period=100 \
    data=cifar10 \
    data.data_root='your_dataset_directory' \
    model=resnet50_mixup model.num_classes=200 \
    loss=mrl or mndcg \
    hydra.run.dir='your_bestmodel_directory' \
    test.checkpoint=best.pth \
```

</p>
</details>

## Bibtex
```
@inproceedings{noh2023rankmixup,
  title={RankMixup: Ranking-Based Mixup Training for Network Calibration},
  author={Noh, Jongyoun and Park, Hyekang and Lee, Junghyup and Ham, Bumsub},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1358--1368},
  year={2023}
}
```
## References
Our code is mainly based on [`FLSD`](https://github.com/torrvision/focal_calibration) and [`MbLS`](https://github.com/by-liu/MbLS/tree/main). For long-tailed(LT) datasets, we borrow the codes from [`MisLAS`](https://github.com/dvlab-research/MiSLAS). Thanks to the authors!

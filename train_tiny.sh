export CUDA_VISIBLE_DEVICES=0

NGPUS=1
CFG_DIR=configs
LOSS_NAME=mrl #mndcg  

# tiny-imagenet (for other losses)
# python tools/train_net.py log_period=100 data=tiny_imagenet model=resnet50_tiny model.num_classes=200 loss=$LOSS_NAME optim=sgd optim.lr=0.1 optim.momentum=0.9 scheduler=multi_step scheduler.milestones="[40, 60]" train.max_epoch=100

# tiny imagenet rankmixup
python tools/train_net.py log_period=100 data=tiny_imagenet model=resnet50_mixup_tiny model.num_classes=200 loss=$LOSS_NAME optim=sgd optim.lr=0.1 optim.momentum=0.9 scheduler=multi_step scheduler.milestones="[40, 60]" train.max_epoch=100

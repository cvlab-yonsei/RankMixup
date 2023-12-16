export CUDA_VISIBLE_DEVICES=0

NGPUS=1
CFG_DIR=configs
LOSS_NAME=mrl #mndcg  

# cifar10 (for other losses)
# python /workspace/work/AdaptiveMargin/tools/train_net.py log_period=100 data=cifar10 model=resnet50_cifar model.num_classes=10 loss=$LOSS_NAME optim=sgd optim.lr=0.1 optim.momentum=0.9 scheduler=multi_step scheduler.milestones="[80, 120]" train.max_epoch=200
# cifar10 rankmixup
python /workspace/work/AdaptiveMargin/tools/train_net.py log_period=100 data=cifar10 model=resnet50_mixup model.num_classes=10 loss=$LOSS_NAME optim=sgd optim.lr=0.1 optim.momentum=0.9 scheduler=multi_step scheduler.milestones="[80, 120]" train.max_epoch=200 

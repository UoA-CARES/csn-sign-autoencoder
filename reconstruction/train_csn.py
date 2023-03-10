import os
import torch
import torch.nn as nn
import wandb
import numpy as np

from torchvision import transforms
from mmcv_csn import ResNet3dCSN
from csn import csn50
from i3d_head import I3DHead
from cls_autoencoder import EncoderDecoder
from reconstruction_head import RecontructionHead
from scheduler import GradualWarmupScheduler
from mmaction.datasets import build_dataset

os.chdir('../')

wandb.init(entity="cares", project="autoencoder",
           group="wlasl-100", name="reconstruction")

# Set up device agnostic code
try:
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
except:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set up dataset
batch_size = 2
train_cfg=dict(
    type='RawframeDataset',
    ann_file='data/wlasl/train_annotations.txt',
    data_prefix='data/wlasl/rawframes',
    pipeline=[
        dict(
            type='SampleFrames',
            clip_len=32,
            frame_interval=2,
            num_clips=1),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='RandomResizedCrop'),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        dict(type='Flip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_bgr=False),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label'])
    ])


test_cfg=dict(
        type='RawframeDataset',
        ann_file='data/wlasl/test_annotations.txt',
        data_prefix='data/wlasl/rawframes',
        pipeline=[
            dict(
                type='SampleFrames',
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=224),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
    ])

work_dir = 'work_dirs/wlasl-dataset/'


os.makedirs(work_dir, exist_ok=True)

# Building the datasets
train_dataset = build_dataset(train_cfg)
test_dataset = build_dataset(test_cfg)

# Setting up dataloaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=4,
                                    pin_memory=True)

# Create a CSN model
encoder = ResNet3dCSN(
    pretrained2d=False,
    # pretrained=None,
    pretrained='https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r50_ig65m_20210617-ce545a37.pth',
    depth=50,
    with_pool2=False,
    bottleneck_mode='ir',
    norm_eval=True,
    zero_init_residual=False,
    bn_frozen=True
)

encoder.init_weights()

reconstruct_head = RecontructionHead()

decoder = I3DHead(num_classes=400,
                 in_channels=2048,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01)

decoder.init_weights()

model = EncoderDecoder(encoder, decoder, reconstruct_head)

# Specify optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.000125, momentum=0.9, weight_decay=0.00001)

# Specify Loss
loss_cls = nn.CrossEntropyLoss()
loss_reconstruct = nn.MSELoss()

# Specify total epochs
epochs = 100

# Specify learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=120, gamma=0.1)

scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[34, 84], gamma=0.1)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=16, after_scheduler=scheduler_steplr)

model.to(device)

# Setup wandb
wandb.watch(model, log_freq=10)

def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.
    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).
    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = np.zeros(len(topk))
    labels = np.array(labels)[:, np.newaxis]
    for i, k in enumerate(topk):
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res[i] = topk_acc_score

    return res


def train_one_epoch(epoch_index, interval=5):
    """Run one epoch for training.
    Args:
        epoch_index (int): Current epoch.
        interval (int): Frequency at which to print logs.
    Returns:
        last_loss (float): Loss value for the last batch.
    """
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, x in enumerate(train_loader):
        # Every data instance is an input + label pair
        images, targets = x['imgs'].to(device), x['label'].to(device)
        images = images.reshape((-1, ) + images.shape[2:])
        targets = targets.reshape(-1, )
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        cls_score, reconstructed = model(images)

        # Get losses
        loss_cls_score = loss_cls(cls_score, targets)
        loss_reconstruct_score = loss_reconstruct(reconstructed, images)
        loss = 0.8 * loss_cls_score + 0.2 * loss_reconstruct_score
        
        # Compute the loss and its gradients
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=40, norm_type=2.0)

        # Adjust learning weights
        optimizer.step()
        

        # Gather data and report
        running_loss += loss.item()
        if i % interval == interval-1:
            last_loss = running_loss / interval  # loss per batch
            print(
                f'Epoch [{epoch_index}][{i+1}/{len(train_loader)}], loss_cls: {loss_cls_score.item():.5}, reconstruct_loss: {loss_reconstruct_score.item():.5} lr: {scheduler.get_last_lr()[0]:.5e}, loss: {last_loss:.5}')
            running_loss = 0.

    return last_loss, scheduler.get_last_lr()[0]


def validate():
    """Run one epoch for validation.
    Returns:
        avg_vloss (float): Validation loss value for the last batch.
        top1_acc (float): Top-1 accuracy in decimal.
        top5_acc (float): Top-5 accuracy in decimal.
    """
    running_vloss = 0.0
    running_vacc = np.zeros(2)

    print('Evaluating top_k_accuracy...')

    with torch.inference_mode():
        for i, x in enumerate(test_loader):
            vimages, vtargets = x['imgs'].to(device), x['label'].to(device)
            vimages = vimages.reshape((-1, ) + vimages.shape[2:])
            vtargets = vtargets.reshape(-1, )
            
            cls_score, reconstructed = model(vimages)

            # Get losses
            loss_cls_score = loss_cls(cls_score, vtargets)
            loss_reconstruct_score = loss_reconstruct(reconstructed, vimages)
            vloss = 0.8 * loss_cls_score + 0.2 * loss_reconstruct_score
            
            running_vloss += vloss

            running_vacc += top_k_accuracy(cls_score.detach().cpu().numpy(),
                                           vtargets.detach().cpu().numpy(), topk=(1, 5))

    avg_vloss = running_vloss / (i + 1)

    acc = running_vacc/len(test_loader)
    top1_acc = acc[0].item()
    top5_acc = acc[1].item()

    return (avg_vloss, top1_acc, top5_acc)


# Train Loop
best_vloss = 1_000_000.

for epoch in range(epochs):
    # Turn on gradient tracking and do a forward pass
    model.train(True)
    avg_loss, learning_rate = train_one_epoch(epoch+1)

    # Turn off  gradients for reporting
    model.train(False)

    avg_vloss, top1_acc, top5_acc = validate()

    print(
        f'top1_acc: {top1_acc:.4}, top5_acc: {top5_acc:.4}, train_loss: {avg_loss:.5}, val_loss: {avg_vloss:.5}')

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = work_dir + f'epoch_{epoch+1}.pth'
        print(f'Saving checkpoint at {epoch+1} epochs...')
        torch.save(model.state_dict(), model_path)

     # Adjust learning rate
    scheduler.step()

    # Track wandb
    wandb.log({'train/loss': avg_loss,
               'train/learning_rate': learning_rate,
               'val/loss': avg_vloss,
               'val/top1_accuracy': top1_acc,
               'val/top5_accuracy': top5_acc})
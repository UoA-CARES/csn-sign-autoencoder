import os
import torch
import torch.nn as nn
import wandb
import numpy as np

from torchvision import transforms
from custom_dataset import VideoFrameDataset, ImglistToTensor
from mmcv_csn import ResNet3dCSN
from i3d_head import I3DHead
from cls_autoencoder import EncoderDecoder
from scheduler import GradualWarmupScheduler

wandb.init(entity="cares", project="autoencoder",
           group="wlasl-10", name="mmcv")

# Set up device agnostic code
try:
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
except:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configs
data_root = os.path.join(os.getcwd(), 'data/wlasl/rawframes') 
ann_file_train = os.path.join(os.getcwd(), 'data/wlasl/train_annotations.txt') 
ann_file_test = os.path.join(os.getcwd(), 'data/wlasl/test_annotations.txt')
work_dir = 'work_dirs/wlasl-500e/'
batch_size = 8

os.makedirs(work_dir, exist_ok=True)


# Setting up data augments
train_pipeline = transforms.Compose([
        ImglistToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((256, 256), scale=(0.6, 1.0)),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    ])

test_pipeline = transforms.Compose([
        ImglistToTensor(), # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize((256, 256)),  # image batch, resize smaller edge to 256
        transforms.CenterCrop((224, 224)),  # image batch, center crop to square 224x224
        transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    ])

# Setting up datasets
train_dataset = VideoFrameDataset(
    root_path=data_root,
    annotationfile_path=ann_file_train,
    clip_len=32,
    frame_interval=2,
    num_clips=1,
    imagefile_template='img_{:05d}.jpg',
    transform=train_pipeline,
    test_mode=False
)

test_dataset = VideoFrameDataset(
    root_path=data_root,
    annotationfile_path=ann_file_test,
    clip_len=32,
    frame_interval=2,
    num_clips=1,
    imagefile_template='img_{:05d}.jpg',
    transform=test_pipeline,
    test_mode=True
)

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


# set up model, loss, optimizer and scheduler
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

decoder = I3DHead(num_classes=400,
                 in_channels=2048,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01)

decoder.init_weights()

model = EncoderDecoder(encoder, decoder)

# # Load model checkpoint
# checkpoint = torch.load(work_dir+'latest.pth')
# model.load_state_dict(checkpoint)


# Specify optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.0000125, momentum=0.9, weight_decay=0.00001)

# Specify Loss
loss_fn = nn.CrossEntropyLoss()

# Specify total epochs
epochs = 150

# Specify learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=120, gamma=0.1)

scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=100, after_scheduler=scheduler_steplr)

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
    for i, (images, targets) in enumerate(train_loader):
        # Every data instance is an input + label pair
        images, targets = images.to(device).permute(
            0, 2, 1, 3, 4), targets.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(images)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, targets)
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
                f'Epoch [{epoch_index}][{i+1}/{len(train_loader)}], lr: {scheduler.get_last_lr()[0]:.5e}, loss: {last_loss:.5}')
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
        for i, (vimages, vtargets) in enumerate(test_loader):
            vimages, vtargets = vimages.to(device), vtargets.to(device)

            voutputs = model(vimages.permute(0, 2, 1, 3, 4))

            vloss = loss_fn(voutputs, vtargets)
            running_vloss += vloss

            running_vacc += top_k_accuracy(voutputs.detach().cpu().numpy(),
                                           vtargets.detach().cpu().numpy(), topk=(1, 5))

    avg_vloss = running_vloss / (i + 1)

    acc = running_vacc/len(test_loader)
    top1_acc = acc[0].item()
    top5_acc = acc[1].item()

    return (avg_vloss, top1_acc, top5_acc)


# Train Loop
best_vloss = 1_000_000.

# Transfer model to device
model.to(device)

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

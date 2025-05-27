import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.losses import DiceLoss
from torch.optim import Adam
from dataset import VSDataset
from model import DynUNet
from loss import DiceWeightedBCELoss
from utils.utils import custom_collate
from loss import DiceWeightedBCELoss
from torch.optim.lr_scheduler import StepLR  
from monai.transforms import (
    Compose, NormalizeIntensityd, RandBiasFieldd,
    RandFlipd, RandRotate90d, RandGaussianNoised, RandZoomd,
    RandAdjustContrastd, RandShiftIntensityd, RandGaussianSmoothd,
    ResizeWithPadOrCropd, ToTensord
)

import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

batch_size = 1
num_workers = 0
pin_memory = False
LEARNING_RATE = 1e-4
num_epochs = 10

def main():

    image_size = (128, 128, 128)

    transform = Compose([
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),  # Z-normalization
        
        # Spatial augmentations
        RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.3),
        RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.3),
        RandRotate90d(keys=["image", "mask"], prob=0.3, max_k=3, spatial_axes= (0,1)),

        # Intensity augmentations
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),
        RandBiasFieldd(keys=["image"], prob=0.3, coeff_range=(0.1, 0.3), degree=3),
        RandGaussianSmoothd(keys=["image"], prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7, 1.5)),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),  
        
        # Resize or pad/crop to final size
        ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=image_size, mode='constant'),
        ToTensord(keys=["image", "mask"]),
    ])

    data_dir= r'D:\VSdata'
    csv_path= r"C:\Users\Acer\Desktop\vs_paths1.csv"
    dataset = VSDataset(
        csv_path= csv_path,
        data_dir=data_dir,
        transform=transform,        
    )


    train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device= torch.device('cpu')
    model= DynUNet(spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2, 2],
        res_block=True,
    )
    
    loss_fn = DiceLoss(to_onehot_y=False, softmax=False, sigmoid=True)
    # loss_function = DiceCELoss(
    #     include_background=True,  # include class 0 (background) in Dice
    #     to_onehot_y=False,        # no need to one-hot encode masks for binary
    #     softmax=False,            # assume logits (no softmax) for binary
    #     sigmoid=True              # apply sigmoid for binary classification
    # )

    # loss_function = DiceWeightedBCELoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    pos_weight = torch.tensor([3.0]).to(device)
    loss_function = DiceWeightedBCELoss(dice_weight=1.0, bce_weight=1.0, pos_weight=pos_weight)
    # loss = loss_function(outputs, labels)
    scaler = torch.cuda.amp.GradScaler()
    # OneCycleLR - this schedular can also be considered
    # scheduler = OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=num_epochs)

    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5) 

    for epoch in range(1):
        model.train()
        running_loss = 0.0
        running_dice_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            print(inputs.shape, targets.shape)
            
            optimizer.zero_grad()

            # outputs = model(inputs)
            # outputs = torch.sigmoid(outputs)
            # dice_loss = loss_fn(outputs, targets)
            # loss = loss_function(outputs, targets)
            # loss.backward()
            # optimizer.step()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                dice_loss = loss_fn(outputs, targets)
                loss = loss_function(outputs, targets)
            
            # Backpropagation with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            running_dice_loss += dice_loss.item()
            
            # if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Dice Loss: {dice_loss.item()}")

        avg_loss = running_loss / len(train_loader)
        avg_dice_loss = running_dice_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Dice Loss: {avg_dice_loss}")

        # scheduler.step()
        
        # Save model checkpoint
        checkpoint_path = f"model_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()
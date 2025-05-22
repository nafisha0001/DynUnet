import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss
from monai.losses import DiceLoss
from torch.optim import Adam
from dataset import VSDataset
from model import DynUNet
import os
import SimpleITK as sitk

from utils.utils import custom_collate
# from loss import DiceWeightedBCELoss

sitk.ProcessObject_SetGlobalWarningDisplay(False)

batch_size = 1
num_workers = 6
pin_memory = True

def dice_per_class(preds, targets, epsilon=1e-6):
    """
    preds, targets: Tensors of shape [B, C, D, H, W] (C=2: background, tumor)
    Returns per-class Dice for each class
    """
    assert preds.shape == targets.shape
    num_classes = preds.shape[1]
    dice_scores = []

    for c in range(num_classes):
        pred_flat = preds[:, c].contiguous().view(-1)
        target_flat = targets[:, c].contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice.item())

    return dice_scores 

def test():
    checkpointPath = "/home/omen/Documents/Nafisha/VS/DynUnetCheckpoints/version5.3.pth"
    device= torch.device('cpu')
    model= DynUNet(spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2, 2],
        res_block=True,
    ).to(device)

    model.load_state_dict(torch.load(checkpointPath, map_location=device))

def main():

    image_size = 128

    slice_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ToTensorV2()
    ])

    # csv_path = r'/home/omen/Documents/VS_data/VS_test_csv.csv'
    # data_dir= r'\home\omen\Documents\VS_data'

    csv_path = "/home/omen/Documents/Nafisha/VS/testingData/final_dataset.csv"
    data_dir = '/home/omen/Documents/Nafisha/VS/testingData'

    dataset = VSDataset(
        csv_path= csv_path,
        data_dir=data_dir,
        transform=slice_transform,           
    )


    test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn= custom_collate
        )

    device= torch.device('cuda')
    # device= 'cpu'
    model= DynUNet(spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2, 2, 2],
        res_block=True,
    ).to(device)

    checkpointPath = "/home/omen/Documents/Nafisha/VS/DynUnet/model_checkpoint.pth"

    model.load_state_dict(torch.load(checkpointPath, map_location=device))

    
    loss_fn = DiceLoss(to_onehot_y=False, softmax=False)
    # loss_function = DiceCELoss(to_onehot_y=False, sigmoid=False)

    scaler = torch.cuda.amp.GradScaler()

    # for epoch in range(num_epochs):
    # model.eval()
    # running_dice_loss= 0.0
    # running_loss = 0.0
    
    # for batch_idx, (inputs, targets) in enumerate(test_loader):
    #     inputs = inputs.to(device)
    #     targets = targets.to(device)
        
    #     with torch.cuda.amp.autocast():
    #         outputs = model(inputs)
    #         outputs= torch.sigmoid(outputs)
    #         dice_loss= loss_fn(outputs, targets)
    #         # loss = loss_function(outputs, targets)
        
    #     running_dice_loss += dice_loss.item()
    #     # running_loss += loss.item()
        
    #     # Loss after 10 batches
    #     # if batch_idx % 10 == 0:
    #     print(f"Step [{batch_idx+1}/{len(test_loader)}], Loss: {dice_loss.item()}")

    # avg_dice_loss = running_dice_loss / len(test_loader)
    # # avg_loss = running_loss / len(test_loader)
    # print(f"Average DIce Loss: {avg_dice_loss}")
    # # print(f"Average DIceCE Loss: {avg_loss}")

    # print("Inference complete!")

    model.eval()
    running_dice_class = torch.zeros(2, device=device)  # [background, tumor]
    num_batches = 0

    scaler = torch.cuda.amp.GradScaler()

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        targets = targets.to(device).long()  

        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            preds = (outputs > 0.5).long()  

        preds_onehot = torch.cat([
            (preds == 0).float(),  
            (preds == 1).float()   
        ], dim=1)

        targets_onehot = torch.cat([
            (targets == 0).float(),
            (targets == 1).float()
        ], dim=1)

        per_class_dice = dice_per_class(preds_onehot, targets_onehot)
        running_dice_class += torch.tensor(per_class_dice, device=device)
        num_batches += 1

        print(f"Step [{batch_idx+1}/{len(test_loader)}], Dice (BG, Tumor): {per_class_dice}")

    avg_dice_class = running_dice_class / num_batches
    print(f"\nAverage Dice - Background: {avg_dice_class[0]:.4f}, Tumor: {avg_dice_class[1]:.4f}")
    print("Inference complete!")

if __name__ == "__main__":
    main()
    # test()
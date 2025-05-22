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
# from torch.utils.data import Subset
from utils.utils import custom_collate
from loss import DiceWeightedBCELoss
from torch.optim.lr_scheduler import StepLR  

import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)

batch_size = 1
num_workers = 0
pin_memory = False
LEARNING_RATE = 1e-4
num_epochs = 10

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

    data_dir= r'D:\VSdata'
    csv_path= r"C:\Users\Acer\Desktop\vs_paths1.csv"
    dataset = VSDataset(
        csv_path= csv_path,
        data_dir=data_dir,
        transform=slice_transform,  
        target_slices=128          
    )


    train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=custom_collate,
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
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5) 

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

        scheduler.step()
        
        # Save model checkpoint
        checkpoint_path = f"model_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()
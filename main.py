import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from Utils.dataloader import get_dataloaders
from Models.model import bulid_model
from train import fit

if __name__ == '__main__':
    wandb.init(project="DL_temaPJ_SKIN", name = "ResNet18-top3ckpt")

    train_dir = r'./Data/Training/01_Source_Data'
    val_dir = r'./Data/Validation/01_Source_Data'
    batch_size = 32
    train_loader, val_loader = get_dataloaders(train_dir, val_dir, batch_size, num_workers =4)

    model = bulid_model(num_classes=15)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)

    top_checkpoints = fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=15, checkpoint_dir='checkpoints')

    print(top_checkpoints)    
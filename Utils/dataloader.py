from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(train_dir, val_dir, batch_size, num_workers=4):
    #transform 정의
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                            std = [0.229,0.224, 0.225])
    ])

    #ImageFolder로 데이터 불러오기
    train_dataset = datasets.ImageFolder(root=train_dir, transform = transform)


    #Dataloder 만들기
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=num_workers)
    val_dataset = datasets.ImageFolder(root=val_dir, transform = transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader 
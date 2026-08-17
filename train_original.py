import torch
from torchvision import datasets, transforms

from torch.utils.data import DataLoader

#transform 정의
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                         std = [0.229,0.224, 0.225])
])

#ImageFolder로 데이터 불러오기
train_dir = r'./Data/Training/01_Source_Data'
train_dataset = datasets.ImageFolder(root=train_dir, transform = transform)

print(train_dataset.classes)
print(len(train_dataset))

print(train_dataset[0])

#Dataloder 만들기
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
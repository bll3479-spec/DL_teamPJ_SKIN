import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models.model import bulid_model


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

# print(train_dataset.classes)
# print(len(train_dataset))
# print(train_dataset[0])

#Dataloder 만들기
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

images, labels = next(iter(train_loader))
#print(images.shape, labels.shape)

model = bulid_model(num_classes=15)
#print(model)

#손실함수, 최적화 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

#학습 루프 
num_epochs= 1

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader):
        #1. 기울기 초기화
        optimizer.zero_grad()
        #2. 순전파
        outputs = model(images)
        #3. 손실 계산
        loss = criterion(outputs, labels)
        #4. 역전파
        loss.backward()
        #5. 가중치 업데이트
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss : {running_loss/len(train_loader):.4f}')

#검증 루프
val_dir = r'./Data/Validation/01_Source_Data'
val_dataset = datasets.ImageFolder(root=val_dir, transform = transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(len(val_dataset))

model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in tqdm(val_loader):
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = correct / total
print(f'Validation Accuracy: {accuracy*100:.2f}%')
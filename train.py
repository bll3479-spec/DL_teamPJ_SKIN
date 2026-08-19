import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import wandb

from evaluate import evaluate


#학습 파이프라인 구축
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader):
        #이미지, 라벨 -> device
        images, labels = images.to(device), labels.to(device)
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
    return running_loss/len(train_loader)

#훈련 과정 및 결과 기록
def fit(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, checkpoint_dir):
    #가중치 폴더 설정
    os.makedirs(checkpoint_dir, exist_ok=True)
    #최고 val_accuracy 추적하며 갱신 위함
    top_checkpoints = []
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}, Loss : {avg_loss}')
        wandb.log({"train_loss": avg_loss, "epoch": epoch+1})
        accuracy, f1 = evaluate(model, val_loader, device, epoch, top_checkpoints, checkpoint_dir)
    return top_checkpoints
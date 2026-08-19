import torch, os
from tqdm import tqdm
import wandb
from sklearn.metrics import f1_score, accuracy_score

def evaluate(model, val_loader, device, epoch, top_checkpoints, checkpoint_dir):
#검증 루프
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f'Validation Accuracy: {accuracy*100: .2f}%, F1(macro): {f1: .4f}')
    wandb.log({"val_accuracy": accuracy, "val_f1":f1})

    if len(top_checkpoints) < 3 or f1 > min(top_checkpoints)[0]:
        filepath = f'{checkpoint_dir}/resnet18_epoch{epoch+1}_f1{f1:.4f}_acc{accuracy:.4f}.pth'
        torch.save(model.state_dict(), filepath)
        top_checkpoints.append((f1, filepath))
        top_checkpoints.sort(key=lambda x: x[0], reverse=True)

        if len(top_checkpoints)>3:
            worst = top_checkpoints.pop()   #정렬 후 마지막 = 가장 낮은 accuracy
            os.remove(worst[1])

    return accuracy, f1
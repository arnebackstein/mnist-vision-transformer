import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    train_pbar = tqdm(train_loader, desc='Training')
    
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Accuracy': f'{100.*correct/total:.2f}%'
        })
    
    return train_loss / len(train_loader), 100.*correct/total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    test_pbar = tqdm(test_loader, desc='Testing')
    
    with torch.no_grad():
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            test_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Accuracy': f'{100.*correct/total:.2f}%'
            })
    
    return test_loss / len(test_loader), 100.*correct/total

def train_model(model, train_loader, test_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    os.makedirs('models', exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        print(f'\nEpoch {epoch} Summary:')
        print(f'Average Train Loss: {train_loss:.4f}')
        print(f'Average Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_acc:.2f}%\n')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }, f'models/checkpoint_epoch_{epoch}.pt')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_train_loss': train_loss,
        'final_test_loss': test_loss,
        'final_test_accuracy': test_acc
    }, 'models/final_model.pt')
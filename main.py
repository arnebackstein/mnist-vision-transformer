from model import VisionTransformer
from data import get_mnist_data, get_device
from train import train_model

def main():
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001
    
    device = get_device()
    print(f'Using device: {device}')
    
    train_loader, test_loader = get_mnist_data(batch_size)
    
    model = VisionTransformer(
        image_size=28,
        patch_size=2,
        num_classes=10,
        dim=32,
        depth=4,
        heads=4,
        mlp_dim=64,
        channels=1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    train_model(model, train_loader, test_loader, num_epochs, learning_rate, device)

if __name__ == '__main__':
    main() 
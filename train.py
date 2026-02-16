"""
Training Script for Multi-Hazard Disaster Detection
Includes training loop, validation, and model checkpointing
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import create_model
from data_loader import create_dataloaders
from edge_module import EdgeExtractor
from utils import (calculate_iou, calculate_pixel_accuracy, 
                  save_checkpoint, CLASS_NAMES)


class Trainer:
    """Training manager for multi-hazard segmentation"""
    
    def __init__(self, model, train_loader, val_loader, 
                 criterion, optimizer, device, edge_extractor=None):
        """
        Args:
            model: Segmentation model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device (cuda/cpu)
            edge_extractor: Edge extraction module
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.edge_extractor = edge_extractor
        
        self.train_losses = []
        self.val_losses = []
        self.val_ious = []
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Apply edge extraction if using edge-aware model
            if self.edge_extractor and images.shape[1] == 3:
                batch_edges = []
                for img in images:
                    # Denormalize for edge detection
                    img_np = img.cpu().numpy()
                    img_np = np.transpose(img_np, (1, 2, 0))
                    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
                             np.array([0.485, 0.456, 0.406]))
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # Extract edges and fuse
                    fused = self.edge_extractor(img_np)
                    batch_edges.append(fused)
                
                images = torch.stack(batch_edges).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        epoch_loss = 0.0
        all_ious = []
        all_accuracies = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        
        with torch.no_grad():
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Apply edge extraction if needed
                if self.edge_extractor and images.shape[1] == 3:
                    batch_edges = []
                    for img in images:
                        img_np = img.cpu().numpy()
                        img_np = np.transpose(img_np, (1, 2, 0))
                        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + 
                                 np.array([0.485, 0.456, 0.406]))
                        img_np = (img_np * 255).astype(np.uint8)
                        
                        fused = self.edge_extractor(img_np)
                        batch_edges.append(fused)
                    
                    images = torch.stack(batch_edges).to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                
                # Calculate metrics
                _, mean_iou = calculate_iou(preds, masks)
                accuracy = calculate_pixel_accuracy(preds, masks)
                
                epoch_loss += loss.item()
                all_ious.append(mean_iou)
                all_accuracies.append(accuracy)
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mIoU': mean_iou,
                    'acc': accuracy
                })
        
        avg_loss = epoch_loss / len(self.val_loader)
        avg_iou = np.mean(all_ious)
        avg_acc = np.mean(all_accuracies)
        
        self.val_losses.append(avg_loss)
        self.val_ious.append(avg_iou)
        
        return avg_loss, avg_iou, avg_acc
    
    def train(self, num_epochs, save_dir='checkpoints'):
        """
        Complete training loop
        
        Args:
            num_epochs (int): Number of epochs to train
            save_dir (str): Directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        
        best_iou = 0.0
        
        print("=" * 70)
        print("Starting Training")
        print("=" * 70)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_iou, val_acc = self.validate(epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val mIoU: {val_iou:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(self.model, self.optimizer, epoch, val_loss, checkpoint_path)
            
            # Save best model
            if val_iou > best_iou:
                best_iou = val_iou
                best_path = os.path.join(save_dir, 'best_model.pth')
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, best_path)
                print(f"  ✓ New best model saved! (mIoU: {best_iou:.4f})")
            
            print("-" * 70)
        
        print("\nTraining completed!")
        print(f"Best validation mIoU: {best_iou:.4f}")


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'train_img_dir': 'data/train/images',
        'train_mask_dir': 'data/train/masks',
        'val_img_dir': 'data/val/images',
        'val_mask_dir': 'data/val/masks',
        'num_classes': 4,
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'img_size': (256, 256),
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints'
    }
    
    print("Multi-Hazard Disaster Detection - Training")
    print("=" * 70)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Number of epochs: {config['num_epochs']}")
    print(f"Image size: {config['img_size']}")
    print("=" * 70)
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_dataloaders(
        config['train_img_dir'],
        config['train_mask_dir'],
        config['val_img_dir'],
        config['val_mask_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers']
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        num_classes=config['num_classes'],
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=4
    )
    
    # Create edge extractor
    edge_extractor = EdgeExtractor()
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config['device'],
        edge_extractor=edge_extractor
    )
    
    # Train
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir']
    )


if __name__ == "__main__":
    main()

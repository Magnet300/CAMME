import os
import torch
import torchvision
from transformers import ViTFeatureExtractor 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch
import random
import open_clip
from PIL import Image, ImageFile
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
import torch_dct as dct
from cross_attention_transformer import TransformerModel
from torchvision.transforms.functional import InterpolationMode
import io
from io import BytesIO

from torchinfo import summary

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
   
    print(f"Global seeds set to: {seed}")

device = "cuda:2" if torch.cuda.is_available() else "cpu"

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To prevent errors from incomplete files
Image.MAX_IMAGE_PIXELS = None  # Allow loading images without a size limit

class AddGaussianNoise:
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class JPEGCompression:
    def __init__(self, quality_range=(60, 100)):
        self.quality_range = quality_range
        
    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class ProbabilisticTransform:
    """Applies a transform with a given probability."""
    def __init__(self, transform, probability=0.1):
        self.transform = transform
        self.probability = probability
        
    def __call__(self, img):
        if random.random() < self.probability:
            return self.transform(img)
        return img

train_transform = transforms.Compose([transforms.RandomResizedCrop(size=(320, 320), scale=(0.9, 1.0), ratio=(0.75, 1.3333),
                                                                   interpolation=InterpolationMode.BICUBIC, antialias=True),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.CenterCrop(320),
                                      ProbabilisticTransform(JPEGCompression(quality_range=(60, 100)), probability=0.1),
                                      ProbabilisticTransform(transforms.GaussianBlur(kernel_size=3), probability=0.1),
                                      ProbabilisticTransform(transforms.ColorJitter(brightness=0.2, contrast=0.2), probability=0.1),
                                      transforms.ToTensor(),
                                      ProbabilisticTransform(AddGaussianNoise(std=0.01), probability=0.1),
                                      transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                           std=[0.26862954, 0.26130258, 0.27577711])])

# Validation transform without augmentation
val_transform = transforms.Compose([transforms.Resize((320, 320), interpolation=InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(320),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                         std=[0.26862954, 0.26130258, 0.27577711])])

class CustomDatasetWithCaptions(torch.utils.data.Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        
        # Changed to case-insensitive extension matching with lower()
        self.real_images = [(os.path.join(real_dir, img), 0) for img in os.listdir(real_dir) 
                           if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        self.fake_images = [(os.path.join(fake_dir, img), 1) for img in os.listdir(fake_dir) 
                           if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                           
        self.all_images = self.real_images + self.fake_images
        
    def __len__(self):
        return len(self.all_images)
        
    def __getitem__(self, idx):
        img_path, label = self.all_images[idx]
        try:
            with Image.open(img_path) as image:
                # Ensure image is in RGB format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Apply transformations (e.g., resizing)
                if self.transform:
                    image = self.transform(image)
                
                # Convert transformed image to grayscale
                grayscale_tensor = transforms.functional.rgb_to_grayscale(image)
                
                # Scale to [-1, 1]
                grayscale_tensor = (grayscale_tensor * 2) - 1
                
                # Apply 2D DCT directly using dct_2d
                DCT_transform = dct.dct_2d(grayscale_tensor, norm='ortho')
        except Exception as e:
            print(f"Error processing image: {img_path}\nException: {e}")
            raise
            
        # Prepare caption
        caption = os.path.splitext(os.path.basename(img_path))[0]
        cleaned_caption = caption.replace('_', ' ')
        
        return image, DCT_transform, label, cleaned_caption

def standard_scale(tensor):
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

class Net(nn.Module):
    def __init__(self, CLIP_model, TransformerModel):
        super(Net, self).__init__()
        self.TransformerModel = TransformerModel(embed_dim=(768), num_heads=8)
        self.CLIP_model = CLIP_model
        
        self.DCT_Embedder = nn.Linear((320*320), 768, bias=False)
        self.relu = torch.nn.ReLU()

    def forward(self, Images, Text_Encodings, DCT_features):
        img_embedding = self.CLIP_model.encode_image(Images)
        text_embedding = self.CLIP_model.encode_text(Text_Encodings)
        
        DCT_features_reshaped = DCT_features.view(DCT_features.size(0), -1)
        DCT_features_reshaped = torch.log(torch.abs(DCT_features_reshaped) + 1e-12)
        DCT_embedding = standard_scale(DCT_features_reshaped)
        
        DCT_embedding = self.relu(self.DCT_Embedder(DCT_embedding))
        
        combined_embedding = torch.stack([img_embedding, DCT_embedding, text_embedding], dim=1)        
        CrossAttention_out, attn_weights = self.TransformerModel(combined_embedding, combined_embedding, combined_embedding)
        
        return CrossAttention_out, attn_weights

def train(CLIP_model, tokenizer, train_ds, val_ds, EPOCHS, BATCH_SIZE, LEARNING_RATE, Weight_Decay, save_path, patience=5, min_delta=0.001):
    model_name = "Midjourney_Model_Using_AdamW_with_augmentation"
    model = Net(CLIP_model, TransformerModel)
    model.to(device)
    
    # Verify the TransformerModel instance
    print("\n=== TransformerModel Verification ===")
    print(f"TransformerModel type: {type(model.TransformerModel)}")
    print(f"TransformerModel attributes: {dir(model.TransformerModel)}")
    
    # First, let's see what the actual parameter name is
    print("\nAll parameter names in model:")
    for name, _ in model.named_parameters():
        print(name)
    
    params = []
    for name, p in model.named_parameters():
        if ("TransformerModel" in name or "DCT_Embedder" in name):  
            params.append(p)
            print(f"\nParameter {name} will be trained")
        else:
            p.requires_grad = False
    
    #optimizer = torch.optim.Adam(params, lr= 5e-5 , weight_decay= 0.01)
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE,  weight_decay=Weight_Decay, betas=(0.9, 0.999))    # Default beta values
    loss_func = nn.CrossEntropyLoss()
    
    train_loader = DataLoader( train_ds, batch_size=BATCH_SIZE,  shuffle=True,  num_workers=4)
    val_loader = DataLoader( val_ds,  batch_size=BATCH_SIZE,  shuffle=False,   num_workers=4 )
    
    best_val_accuracy = 0.0
    counter = 0
    early_stop = False
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        total_samples = 0
        correct_train_preds = 0
        
        for batch_idx, (image, DCT_features, labels, caption) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            DCT_features = DCT_features.to(device)
            labels = labels.long().to(device)
            Text_Emb = tokenizer(list(caption), context_length=77).to(device)
            
            optimizer.zero_grad()
            logits, _ = model(image, Text_Emb, DCT_features)
            
            loss = loss_func(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_train_preds += torch.sum(preds == labels).item()
            total_samples += labels.size(0)
            
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train_preds / total_samples
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_total_samples = 0
        correct_val_preds = 0
        
        with torch.no_grad():
            for batch_idx, (image, DCT_features, labels, caption) in enumerate(tqdm(val_loader)):
                image = image.to(device)
                DCT_features = DCT_features.to(device)
                labels = labels.long().to(device)
                Text_Emb = tokenizer(list(caption), context_length=77).to(device)
                
                logits, _ = model(image, Text_Emb, DCT_features)
                loss = loss_func(logits, labels)
                
                total_val_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct_val_preds += torch.sum(preds == labels).item()
                val_total_samples += labels.size(0)
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val_preds / val_total_samples
        
        print(f"\nModel: {model_name}")  
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Training Loss: {avg_train_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")
        
        # Save best model and early stopping
        if val_accuracy > best_val_accuracy + min_delta:
            print(f"Validation accuracy improved from {best_val_accuracy:.4f} to {val_accuracy:.4f}")
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            counter = 0
            save_file = f"{model_name}.pth"
            torch.save(model.state_dict(), os.path.join(save_path, save_file))
            print(f"Model saved as: {save_file}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience} for best Validation Accuracy {best_val_accuracy:.4f}")
            
        if counter >= patience:
            print(f"Early stopping triggered! Best validation accuracy: {best_val_accuracy:.4f} at epoch {best_epoch+1}")
            print(f"Best model saved as: {model_name}.pth")
            early_stop = True
            break
            
        print("-" * 60)
        
    return best_val_accuracy, best_epoch + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=24, help='seed value')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay regularization value")
    parser.add_argument("--savepath", type=str, default="./new_checkpoints")
    parser.add_argument("--size", type=str, default="large")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_real_dir", type=str, required=True, help="Path to real training images")
    parser.add_argument("--train_fake_dir", type=str, required=True, help="Path to fake training images")
    parser.add_argument("--val_real_dir", type=str, required=True, help="Path to real validation images")
    parser.add_argument("--val_fake_dir", type=str, required=True, help="Path to fake validation images")
    args = parser.parse_args()
    
    os.makedirs(args.savepath, exist_ok=True)
    set_global_seeds(args.seed)

    EPOCHS = args.epochs
    BATCH_SIZE = 64
    LEARNING_RATE = args.lr 
    Weight_Decay = args.weight_decay
    
    CLIP_model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    
    # Create the dataset
    train_dataset = CustomDatasetWithCaptions(real_dir=args.train_real_dir, fake_dir=args.train_fake_dir, transform=train_transform)
    val_dataset = CustomDatasetWithCaptions(real_dir=args.val_real_dir, fake_dir=args.val_fake_dir, transform=val_transform)
    
    best_accuracy, best_epoch = train(CLIP_model, tokenizer, train_dataset, val_dataset, EPOCHS, BATCH_SIZE, LEARNING_RATE, Weight_Decay, args.savepath)

if __name__ == "__main__":
    main()













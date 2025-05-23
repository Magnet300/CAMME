import os
import torch
import torchvision
from transformers import ViTFeatureExtractor 
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
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
import time  # Added for timing measurements

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
   
    print(f"Global seeds set to: {seed}")

device = "cuda:2" if torch.cuda.is_available() else "cpu"

ImageFile.LOAD_TRUNCATED_IMAGES = True  # To prevent errors from incomplete files
Image.MAX_IMAGE_PIXELS = None  # Allow loading images without a size limit


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

# Validation transform without augmentation
test_transform = transforms.Compose([transforms.Resize((320, 320), interpolation=InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(320),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                         std=[0.26862954, 0.26130258, 0.27577711])])

def count_trainable_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def testdata(CLIP_model, tokenizer, test_dataset, BATCH_SIZE, CLASSES, args):
    print("Number of test samples:", len(test_dataset))
    
    # Initialize model
    model = Net(CLIP_model, TransformerModel)
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    
    # Load model weights directly
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Trained model loaded successfully")
    
    # Calculate and print number of trainable parameters
    total_params = count_trainable_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Calculate parameter size in MB
    param_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Model parameter size: {param_size_mb:.2f} MB")
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Evaluation
    model.eval()
    total_test_loss = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    
    # Lists to store all predictions and labels
    all_preds = []
    all_labels = []
    
    # Timing variables
    total_inference_time = 0
    total_samples = 0
    
    # Warmup run to eliminate initialization overhead
    with torch.no_grad():
        for i, (image, DCT_features, labels, caption) in enumerate(test_loader):
            if i >= 3:  # Just do a few batches for warmup
                break
            image = image.to(device)
            DCT_features = DCT_features.to(device)
            Text_Emb = tokenizer(list(caption), context_length=77).to(device)
            _ = model(image, Text_Emb, DCT_features)
    
    # Actual timing run
    with torch.no_grad():
        for batch_idx, (image, DCT_features, labels, caption) in enumerate(tqdm(test_loader)): 
            # Move data to device
            image = image.to(device)
            DCT_features = DCT_features.to(device)
            labels = labels.to(device)
            Text_Emb = tokenizer(list(caption), context_length=77).to(device)
            
            batch_size = image.size(0)
            total_samples += batch_size
            
            # Start timing
            if device.startswith('cuda'):
                torch.cuda.synchronize()  # Make sure GPU operations complete before timing
            start_time = time.time()
            
            # Forward pass
            logits, _ = model(image, Text_Emb, DCT_features)
            
            # End timing
            if device.startswith('cuda'):
                torch.cuda.synchronize()  # Make sure GPU operations complete before stopping timer
            end_time = time.time()
            
            # Accumulate inference time
            batch_inference_time = end_time - start_time
            total_inference_time += batch_inference_time
            
            # Continue with normal evaluation
            loss = loss_func(logits, labels)
            _, pred = torch.max(logits, dim=1)
            
            # Store predictions and labels
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update metrics
            total_test_loss += loss.item()
            # Calculate confusion matrix elements
            tp += ((pred == 0) & (labels == 0)).sum().item()
            tn += ((pred == 1) & (labels == 1)).sum().item()
            fp += ((pred == 0) & (labels == 1)).sum().item()
            fn += ((pred == 1) & (labels == 0)).sum().item()
    
    # Calculate metrics
    avg_test_loss = total_test_loss / len(test_loader)
    
    # Calculate performance metrics
    eps = 1e-8  # small epsilon to avoid division by zero
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Calculate timing metrics
    avg_inference_time_per_sample = total_inference_time / total_samples
    avg_inference_time_per_batch = total_inference_time / len(test_loader)
    samples_per_second = total_samples / total_inference_time
    
    # Print results
    print("\nTest Results:")
    print(f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, accuracy: {accuracy:.4f}")
    print(f"{precision*100:.2f} / {recall*100:.2f} / {f1*100:.2f} / {accuracy*100:.2f}")
    
    print("\nInference Time Metrics:")
    print(f"Total inference time: {total_inference_time:.4f} seconds")
    print(f"Average inference time per sample: {avg_inference_time_per_sample*1000:.4f} ms")
    print(f"Average inference time per batch (size {BATCH_SIZE}): {avg_inference_time_per_batch*1000:.4f} ms")
    print(f"Throughput: {samples_per_second:.2f} samples/second")
    
    return accuracy, precision, recall, f1, avg_inference_time_per_sample

def measure_inference_times(model, test_loader, tokenizer, num_runs=100):
    """Measure inference times more precisely with multiple runs"""
    model.eval()
    inference_times = []
    
    # Get a single batch for repeated testing
    for image, DCT_features, _, caption in test_loader:
        sample_image = image.to(device)
        sample_dct = DCT_features.to(device)
        sample_text = tokenizer(list(caption), context_length=77).to(device)
        break  # Just use the first batch
    
    batch_size = sample_image.size(0)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_image, sample_text, sample_dct)
    
    # Measurement runs
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            start_time = time.time()
            
            _ = model(sample_image, sample_text, sample_dct)
            
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end_time = time.time()
            
            inference_times.append(end_time - start_time)
    
    # Calculate statistics
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    median_time = np.median(inference_times)
    
    print("\nDetailed Inference Time Analysis:")
    print(f"Mean batch inference time ({batch_size} samples): {mean_time*1000:.4f} ms (±{std_time*1000:.4f} ms)")
    print(f"Median batch inference time: {median_time*1000:.4f} ms")
    print(f"Mean per-sample inference time: {(mean_time/batch_size)*1000:.4f} ms")
    print(f"Throughput: {batch_size/mean_time:.2f} samples/second")
    
    return {
        "mean_batch_time": mean_time,
        "std_batch_time": std_time,
        "median_batch_time": median_time,
        "per_sample_time": mean_time/batch_size,
        "throughput": batch_size/mean_time
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=24, help='seed value')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--test_real_dir', type=str, required=True, help="Path to real test images")
    parser.add_argument('--test_fake_dir', type=str, required=True, help="Path to fake test images")
    parser.add_argument('--detailed_timing', action='store_true', help="Run detailed timing analysis")
    args = parser.parse_args()
    
    set_global_seeds(args.seed)
    BATCH_SIZE = 64
    CLASSES = 2
    
    CLIP_model, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup')
    
    test_dataset = CustomDatasetWithCaptions(real_dir=args.test_real_dir, fake_dir=args.test_fake_dir, transform=test_transform)
    
    # Run normal evaluation with timing
    accuracy, precision, recall, f1, avg_time = testdata(CLIP_model, tokenizer, test_dataset, BATCH_SIZE, CLASSES, args)
    
    # Optional: Run more detailed timing analysis
    if args.detailed_timing:
        print("\nRunning detailed timing analysis...")
        model = Net(CLIP_model, TransformerModel)
        model.to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        timing_results = measure_inference_times(model, test_loader, tokenizer)
        
        # Print summary
        print("\nModel Inference Summary:")
        print(f"Model Accuracy: {accuracy*100:.2f}%")
        print(f"Average Inference Time: {avg_time*1000:.4f} ms per sample")
        print(f"Samples Processed per Second: {1/avg_time:.2f}")

if __name__ == "__main__":
    main()
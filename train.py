# 1. Install SMP & Albumentations
!pip install segmentation_models_pytorch -q
!pip install albumentations -q

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler # Mixed Precision for speed
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import zipfile

# ============================================================================
# 1. CONFIGURATION (The "Winning" Hyperparameters)
# ============================================================================
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_W, IMG_H = 512, 256  # Divisible by 32 (Requirement for SegFormer)
    BATCH_SIZE = 8           # Lower to 4 if you get CUDA OOM
    EPOCHS = 25              # SegFormer converges faster, but 35 is safe
    LR = 6e-5                # Transformers need lower LR than ResNet (1e-4 is too high)
    ENCODER = "mit_b2"       # The SegFormer Backbone
    WEIGHTS = "imagenet"
    
    # KAGGLE PATHS - VERIFY THESE!
    TRAIN_IMG = "/kaggle/input/iba-hackathon-training/Offroad_Segmentation_Training_Dataset/train/Color_Images"
    TRAIN_MASK = "/kaggle/input/iba-hackathon-training/Offroad_Segmentation_Training_Dataset/train/Segmentation"
    VAL_IMG = "/kaggle/input/iba-hackathon-training/Offroad_Segmentation_Training_Dataset/val/Color_Images"
    VAL_MASK = "/kaggle/input/iba-hackathon-training/Offroad_Segmentation_Training_Dataset/val/Segmentation"
    
    # FOR SUBMISSION
    TEST_IMG_DIR = '/kaggle/input/iba-hackathon-testing/test_public_80/Color_Images'
    SUBMISSION_DIR = '/kaggle/working/final_submission'

# ============================================================================
# 2. DATASET & AUGMENTATIONS (Heavy Augmentation to fix 0.28 score)
# ============================================================================
# We use strong augmentations to force the model to learn shapes, not just colors.
train_transform = A.Compose([
    A.Resize(Config.IMG_H, Config.IMG_W),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3), # Helps with lighting changes
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(Config.IMG_H, Config.IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class OffroadDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
        self.v_map = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        img = np.array(Image.open(img_path).convert("RGB"))
        m_raw = np.array(Image.open(mask_path)) # Keep original size for now
        
        # Map raw values to 0-9 class IDs
        mask = np.zeros_like(m_raw, dtype=np.uint8)
        for raw, val in self.v_map.items(): mask[m_raw == raw] = val
        
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask'].long()
            
        return img, mask

# ============================================================================
# 3. MODEL SETUP (SegFormer + Custom Loss)
# ============================================================================
# Initialize SegFormer B2
model = smp.Segformer(
    encoder_name=Config.ENCODER,
    encoder_weights=Config.WEIGHTS,
    in_channels=3,
    classes=10
).to(Config.DEVICE)

# Weighted Loss to fix "Rocks" and "Logs" being ignored
# Weights: [Back, Trees, Lush, DryG, DryB, Clutter, Logs, Rocks, Land, Sky]
weights = torch.tensor([1.0, 1.5, 1.5, 1.5, 1.5, 2.0, 8.0, 8.0, 1.0, 1.0]).to(Config.DEVICE)

# Combo Loss: CE for classification + Dice for shape overlap
criterion = lambda p, t: nn.CrossEntropyLoss(weight=weights)(p, t) + smp.losses.DiceLoss(mode='multiclass')(p, t)

optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-2)

# Cosine Scheduler: The secret sauce for Transformers
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-7)

# Scaler for Mixed Precision (Faster training)
scaler = GradScaler()

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================
def run_training():
    train_loader = DataLoader(OffroadDataset(Config.TRAIN_IMG, Config.TRAIN_MASK, train_transform), 
                              batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(OffroadDataset(Config.VAL_IMG, Config.VAL_MASK, val_transform), 
                            batch_size=Config.BATCH_SIZE, num_workers=2)

    best_iou = 0.0
    print(f"🚀 Starting SegFormer-{Config.ENCODER} Training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(Config.DEVICE), masks.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # Mixed Precision Forward Pass
            with autocast():
                outputs = model(imgs)
                # Resize outputs if SegFormer outputs different size (usually 1/4th)
                # SMP SegFormer usually handles this, but explicit check is good
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)

            # Mixed Precision Backward Pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=optimizer.param_groups[0]['lr'])
        
        # Update Scheduler
        scheduler.step()
        
        # Validation
        val_iou = validate(val_loader, model)
        print(f"--> VAL mIoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "/kaggle/working/best_segformer_b2.pth")
            print(f"🔥 NEW BEST: {best_iou:.4f} (Saved)")

def validate(loader, model):
    model.eval()
    tp_l, fp_l, fn_l, tn_l = [], [], [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(Config.DEVICE), masks.to(Config.DEVICE)
            with autocast():
                outputs = model(imgs)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                preds = torch.argmax(outputs, dim=1)
            
            tp, fp, fn, tn = smp.metrics.get_stats(preds, masks, mode='multiclass', num_classes=10)
            tp_l.append(tp); fp_l.append(fp); fn_l.append(fn); tn_l.append(tn)
            
    return smp.metrics.iou_score(torch.cat(tp_l), torch.cat(fp_l), torch.cat(fn_l), torch.cat(tn_l), reduction="macro")

# ============================================================================
# 5. SUBMISSION GENERATOR (With TTA!)
# ============================================================================
def generate_submission():
    print("\ngenerating Submission with TTA (Test Time Augmentation)...")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    
    # Load Best Model
    model.load_state_dict(torch.load("/kaggle/working/best_segformer_b2.pth"))
    model.eval()
    
    test_files = [f for f in os.listdir(Config.TEST_IMG_DIR) if f.endswith(('.png', '.jpg'))]
    
    with torch.no_grad():
        for f in tqdm(test_files, desc="Inference"):
            img_path = os.path.join(Config.TEST_IMG_DIR, f)
            img_pil = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img_pil.size
            
            # Preprocess
            img_t = val_transform(image=np.array(img_pil))['image'].unsqueeze(0).to(Config.DEVICE)
            
            # --- TTA: Predict on Normal + Flipped Image ---
            # 1. Normal
            out1 = model(img_t)
            out1 = nn.functional.interpolate(out1, size=(Config.IMG_H, Config.IMG_W), mode='bilinear')
            
            # 2. Flipped (Horizontal)
            img_flip = torch.flip(img_t, [3])
            out2 = model(img_flip)
            out2 = nn.functional.interpolate(out2, size=(Config.IMG_H, Config.IMG_W), mode='bilinear')
            out2 = torch.flip(out2, [3]) # Flip back
            
            # Average predictions
            final_out = (out1 + out2) / 2.0
            pred_mask = torch.argmax(final_out, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # Resize to Original Size (CRITICAL for score)
            final_img = Image.fromarray(pred_mask).resize((orig_w, orig_h), resample=Image.NEAREST)
            final_img.save(os.path.join(Config.SUBMISSION_DIR, f))
            
    # Zip it
    print("Zipping...")
    !zip -q -r submission_segformer.zip {Config.SUBMISSION_DIR}
    print("✅ Done! Download 'submission_segformer.zip'")

if __name__ == "__main__":
    run_training()
    # Uncomment this line when you want to generate the submission file!
    # generate_submission()

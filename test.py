import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

# ============================================================================
# 1. TARGETED PATH DETECTION (Filtering for the 800 Test Images)
# ============================================================================
print("🔍 Searching for the official 800 test images...")

TEST_IMG_DIR = None
TEST_MASK_DIR = None
MASKS_FOUND = False

# We are specifically looking for the 'test_public_80' subfolder
for root, dirs, files in os.walk('/kaggle/input'):
    if 'test_public_80' in root:
        if 'Color_Images' in dirs:
            TEST_IMG_DIR = os.path.join(root, 'Color_Images')
        if 'Segmentation' in dirs:
            potential_mask_dir = os.path.join(root, 'Segmentation')
            if len(os.listdir(potential_mask_dir)) > 0:
                TEST_MASK_DIR = potential_mask_dir
                MASKS_FOUND = True

if TEST_IMG_DIR:
    # Double check count
    count = len(os.listdir(TEST_IMG_DIR))
    print(f"✅ Found Test Folder: {TEST_IMG_DIR}")
    print(f"📊 Image Count: {count} (Should be around 800)")
else:
    print("❌ Error: Could not find 'test_public_80' folder. Check your dataset names.")
    exit()

# Find the Model
MODEL_PATH = '/kaggle/working/best_segformer_b2.pth'
if not os.path.exists(MODEL_PATH):
    # Search for it if not in default working dir
    for root, dirs, files in os.walk('/kaggle'):
        if 'best_segformer_b2.pth' in files:
            MODEL_PATH = os.path.join(root, 'best_segformer_b2.pth')
            break

# ============================================================================
# 2. SETUP & EVALUATION LOGIC
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_W, IMG_H = 512, 256
OUTPUT_DIR = '/kaggle/working/submission_official'
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = smp.Segformer(encoder_name="mit_b2", in_channels=3, classes=10).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def compute_iou(pred, target):
    ious = []
    for cls in range(10):
        p = pred == cls
        t = target == cls
        intersection = (p & t).sum()
        union = (p | t).sum()
        if union > 0: ious.append(intersection / union)
    return np.nanmean(ious) if ious else 0

test_transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

V_MAP = {0:0, 100:1, 200:2, 300:3, 500:4, 550:5, 700:6, 800:7, 7100:8, 10000:9}

# ============================================================================
# 3. RUN (800 Images Only)
# ============================================================================
test_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg'))]
iou_scores = []

print(f"🚀 Processing official test set...")
with torch.no_grad():
    for f in tqdm(test_files):
        img_path = os.path.join(TEST_IMG_DIR, f)
        img_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img_pil.size
        
        t_img = test_transform(image=np.array(img_pil))['image'].unsqueeze(0).to(DEVICE)
        
        # Inference with TTA
        out1 = model(t_img)
        out1 = nn.functional.interpolate(out1, size=(IMG_H, IMG_W), mode='bilinear')
        out2 = model(torch.flip(t_img, [3]))
        out2 = nn.functional.interpolate(out2, size=(IMG_H, IMG_W), mode='bilinear')
        out2 = torch.flip(out2, [3])
        
        pred_mask = torch.argmax((out1 + out2)/2, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
        if MASKS_FOUND:
            m_path = os.path.join(TEST_MASK_DIR, f)
            if os.path.exists(m_path):
                m_raw = np.array(Image.open(m_path).resize((IMG_W, IMG_H), resample=Image.NEAREST))
                m_true = np.zeros_like(m_raw)
                for raw, val in V_MAP.items(): m_true[m_raw == raw] = val
                iou_scores.append(compute_iou(pred_mask, m_true))

        # Save only the official 800 masks
        Image.fromarray(pred_mask).resize((orig_w, orig_h), resample=Image.NEAREST).save(os.path.join(OUTPUT_DIR, f))

# ============================================================================
# 4. FINAL OUTPUT
# ============================================================================
print("\n" + "="*40)
if ious := iou_scores:
    print(f"🏆 OFFICIAL TEST mIoU: {np.mean(ious):.4f}")
else:
    print("⚠️ No masks in this folder. Submit the zip to see your score.")
print("="*40)

!zip -q -r submission_800.zip {OUTPUT_DIR}
print("✅ Created 'submission_800.zip'. Download and submit this one!")

# IBA Hackathon - Semantic Segmentation (SegFormer MiT-B2)

### 📊 Performance
- **Validation mIoU:** 0.6184
- **Test mIoU:** 0.3323

### 🔗 External File Links
Due to file size limits, the following are hosted on Google Drive:
- **Trained Model (.pth):** https://drive.google.com/drive/folders/1tRzyjD8YnBv6xkmZJ-XtFcLw7QGCG41r
- **Final Submission Masks (.zip):** https://drive.google.com/drive/folders/1tRzyjD8YnBv6xkmZJ-XtFcLw7QGCG41r

### 🛠️ How to Run
1. **Training:** Run `python train.py` (Requires `segmentation_models_pytorch`).
2. **Inference:** Run `python test.py`. This script uses Test-Time Augmentation (TTA) to generate the final masks.

# LSUN_group_project

## 1. Overview

This project focuses on **Supervised Learning** using the LSUN (Large-scale Scene Understanding) dataset.  
(Ujjwal handled the unsupervised models, not included here.)

I selected **6 LSUN scene categories**, each containing **30,000 images**, trainng using **10,000 images**, split as follows:

- **Training:** 7,000
- **Validation:** 1,500
- **Testing:** 1,500

---

## 2. Summary Table

![alt text](image.png)

---

## 3. Download LSUN Dataset

Download raw LSUN scene files from:

**http://dl.yf.io/lsun/scenes/**

Ensure the LMDB files are extracted before running any scripts.

---

## 4. Run the Main Script

Run this script first to train the baseline supervised CNN model:

```bash
python lsun_classification.py
```

---

## 5. Convert .webp to .jpg

If your LSUN images are not JPG files, convert them using:

```bash
python convert_webp_to_jpg.py
```

---

# 6. Confusion Matrix Fix

If an error occurs before generating the confusion matrix but the model is already trained, run:

```bash
python create_confusion_matrix.py
```

---

# 7. Real World Application

Create "test_photos" folder. Put all of your photos which related to features into it, run:

```bash
cd application
python real_world_application.py
```

---

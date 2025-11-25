# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 10:39:29 2025

@author: Eric
"""

import os
import shutil
import time
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0, VGG16, DenseNet121

from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# ---------------------------
# CONFIG
# ---------------------------
DATA_ROOT = Path('E:/lsun_raw')
ORG_DIR = Path('E:/lsun_tra_val')

# IMPROVED SETTINGS FOR BETTER ACCURACY
IMG_SIZE = 224  # Increased from 128 - better for complex scenes
BATCH_SIZE = 32  # Reduced from 64 - more stable gradients
EPOCHS = 50  # Reduced from 100 - with early stopping this is enough
MAX_IMAGES_PER_CLASS = 10000  # Reduced from 30000 - faster iteration, still plenty

EXPECTED_CATEGORIES = ['bedroom', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room']
VALID_IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.webp']

"""
MODEL REQUIREMENTS FOR 6 CLASSES:
- Baseline CNN: 5,000-10,000 images per class
- ResNet50/EfficientNet: 10,000-20,000 images per class  
- DenseNet121: 15,000-25,000 images per class
"""

# ---------------------------
# DATA VALIDATION
# ---------------------------

def count_images_in_dir(p):
    if not p.exists() or not p.is_dir():
        return 0
    cnt = 0
    for ext in VALID_IMAGE_EXTS:
        cnt += len(list(p.rglob(f'*{ext}')))
    return cnt

def validate_and_diagnose_data():
    """Complete data validation before training."""
    print("\n" + "="*80)
    print("DATA VALIDATION & DIAGNOSIS")
    print("="*80 + "\n")
    
    # Check raw data
    print("1. Checking RAW data (E:\\lsun_raw)...")
    if not DATA_ROOT.exists():
        print(f"❌ ERROR: {DATA_ROOT} not found!")
        print("   Create this folder and add your image subfolders.")
        return False
    
    raw_stats = {}
    raw_issues = []
    
    for category in EXPECTED_CATEGORIES:
        cat_path = DATA_ROOT / category
        if not cat_path.exists():
            raw_issues.append(f"Missing folder: {category}")
            continue
        
        img_count = count_images_in_dir(cat_path)
        raw_stats[category] = img_count
        
        if img_count == 0:
            raw_issues.append(f"'{category}' has 0 images")
        elif img_count < 100:
            raw_issues.append(f"'{category}' has only {img_count} images (need 100+)")
        else:
            print(f"   ✓ {category:18}: {img_count:,} images")
    
    if raw_issues:
        print("\n❌ RAW DATA ISSUES:")
        for issue in raw_issues:
            print(f"   - {issue}")
        return False
    
    # Check class balance
    total_raw = sum(raw_stats.values())
    max_count = max(raw_stats.values())
    min_count = min(raw_stats.values())
    imbalance = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n✓ Raw data OK: {len(raw_stats)} categories, {total_raw:,} total images")
    
    if imbalance > 2.0:
        print(f"⚠️  WARNING: Class imbalance detected (ratio: {imbalance:.2f}x)")
        print("   Most common class has 2x+ images vs least common")
        print("   This may hurt model performance!")
    else:
        print(f"✓ Classes are balanced (ratio: {imbalance:.2f}x)")
    
    print()
    
    # Check organized data if exists
    if ORG_DIR.exists():
        print("2. Checking ORGANIZED data (E:\\lsun_tra_val)...")
        splits = ['train', 'val', 'test']
        org_issues = []
        
        for split in splits:
            split_dir = ORG_DIR / split
            for category in EXPECTED_CATEGORIES:
                cat_dir = split_dir / category
                if not cat_dir.exists() or count_images_in_dir(cat_dir) == 0:
                    org_issues.append(f"{split}/{category} missing or empty")
        
        if org_issues:
            print("   ⚠️  ISSUES FOUND:")
            for issue in org_issues:
                print(f"      - {issue}")
            print(f"\n   Solution: Delete {ORG_DIR} and re-run to reorganize\n")
            return False
        else:
            print("   ✓ Organized data structure is valid\n")
    
    print("="*80 + "\n")
    return True

# ---------------------------
# ORGANIZE DATA
# ---------------------------

def organize_dataset_structure(source_dir, target_dir, max_images_per_class=MAX_IMAGES_PER_CLASS):
    """Build train/val/test folders from raw data."""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Check if already organized
    if target_dir.exists():
        splits = ['train', 'val', 'test']
        has_all = True
        for split in splits:
            for category in EXPECTED_CATEGORIES:
                if count_images_in_dir(target_dir / split / category) == 0:
                    has_all = False
                    break
        
        if has_all:
            print(f"✓ Organized data exists at: {target_dir}")
            print("  Skipping reorganization (delete folder to reorganize)\n")
            return str(target_dir)
    
    print(f"\nOrganizing dataset...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}\n")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        (target_dir / split).mkdir(parents=True, exist_ok=True)

    for category in EXPECTED_CATEGORIES:
        cat_src = source_dir / category
        
        # Gather images
        image_files = []
        for ext in VALID_IMAGE_EXTS:
            image_files.extend(list(cat_src.rglob(f'*{ext}')))
        
        print(f"Processing '{category}': found {len(image_files):,} images")
        
        if len(image_files) < 10:
            print(f"   ⚠️  Too few images, skipping (need at least 10)")
            continue
        
        # Limit images
        if len(image_files) > max_images_per_class:
            image_files = random.sample(image_files, max_images_per_class)
            print(f"   Limited to {max_images_per_class:,} images")
        
        # Split data: 70% train, 15% val, 15% test
        train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=SEED, shuffle=True)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=SEED, shuffle=True)
        
        # Copy files
        splits_dict = {'train': train_files, 'val': val_files, 'test': test_files}
        for split_name, files in splits_dict.items():
            split_cat_dir = target_dir / split_name / category
            split_cat_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(files):
                dst = split_cat_dir / f"{category}_{i}{src.suffix}"
                if not dst.exists():
                    shutil.copy2(src, dst)
        
        print(f"   ✓ train={len(train_files):,}, val={len(val_files):,}, test={len(test_files):,}")
    
    print(f"\n✓ Organization complete!\n")
    return str(target_dir)

# ---------------------------
# DATA GENERATORS
# ---------------------------

def create_data_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
        Data generators WITHOUT rescaling for transfer learning models.
        Transfer models need raw [0-255] pixel values.
    """
    # For transfer learning: NO rescaling, keep images in [0-255] range
    train_datagen = ImageDataGenerator(
        rotation_range=20,  
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],  
        zoom_range=0.2,  
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator()

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', shuffle=True, seed=SEED
    )

    val_generator = test_datagen.flow_from_directory(
        val_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )

    print(f"\nData generators created:")
    print(f"  Train: {train_generator.samples:,} samples")
    print(f"  Val:   {val_generator.samples:,} samples")
    print(f"  Test:  {test_generator.samples:,} samples")
    print(f"  Number of classes: {len(train_generator.class_indices)}")
    print(f"  Classes: {list(train_generator.class_indices.keys())}\n")

    return train_generator, val_generator, test_generator

def create_data_generators_for_cnn(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Separate generators for Baseline CNN which needs [0,1] rescaled images.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Baseline CNN needs this!
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.2,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', shuffle=True, seed=SEED
    )
    val_generator = test_datagen.flow_from_directory(
        val_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=(img_size, img_size), batch_size=batch_size,
        class_mode='categorical', shuffle=False
    )

    return train_generator, val_generator, test_generator

# ---------------------------
# MODELS
# ---------------------------

def create_baseline_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=4, l2_reg=0.01):
    """IMPROVED Baseline CNN - Deeper architecture for complex scenes"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.4),
        
        # Block 4
        layers.Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),

        # Dense layers
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='improved_baseline_cnn')
    return model

def create_transfer_learning_model(base_model_name='ResNet50', input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                   num_classes=4, trainable=False):
    """
    Transfer Learning with CORRECT preprocessing for each model.
    Each pre-trained model needs its specific preprocessing.
    """
    base_models = {
        'ResNet50': ResNet50,
        'EfficientNetB0': EfficientNetB0,
        'VGG16': VGG16,
        'DenseNet121': DenseNet121
    }
    
    base_model_class = base_models.get(base_model_name, ResNet50)
    base_model = base_model_class(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = trainable
    
    # NEW: Build model with proper preprocessing layer
    inputs = layers.Input(shape=input_shape)
    
    # Apply model-specific preprocessing
    if base_model_name == 'ResNet50':
        x = resnet_preprocess(inputs)  # Subtracts ImageNet mean
    elif base_model_name == 'EfficientNetB0':
        x = efficientnet_preprocess(inputs)  # Scales to [-1, 1]
    elif base_model_name == 'VGG16':
        x = vgg_preprocess(inputs)  # Subtracts ImageNet mean, BGR format
    elif base_model_name == 'DenseNet121':
        x = densenet_preprocess(inputs)  # Scales to [0, 1] then normalizes
    else:
        x = inputs
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs, name=f'{base_model_name}_transfer')
    return model

# ---------------------------
# TRAIN / EVAL
# ---------------------------

def train_model(model, train_gen, val_gen, epochs=EPOCHS, model_name='model', initial_lr=0.001):
    # IMPROVED callbacks with better learning rate schedule
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,  # Increased patience
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'{model_name}_best.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        )
    ]

    print(f"\n{'='*80}")
    print(f"Training: {model_name}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - start_time
    
    print(f"\n✓ Training finished in {elapsed:.2f}s ({elapsed/60:.1f} min)\n")
    return history, elapsed

def evaluate_model(model, test_gen, model_name='model'):
    print(f"Evaluating {model_name}...")
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    test_gen.reset()
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    class_names = list(test_gen.class_indices.keys())
    
    # Handle missing classes in test set
    unique_classes = sorted(np.unique(y_true))
    if len(unique_classes) < len(class_names):
        print(f"\nWarning: Only {len(unique_classes)} out of {len(class_names)} classes present in test set")
        target_names = [class_names[i] for i in unique_classes]
        labels = unique_classes
    else:
        target_names = class_names
        labels = None
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, 
                                labels=labels,
                                target_names=target_names, 
                                digits=4,
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred_classes, labels=labels)
    # Return 5 values to match your unpacking
    return test_accuracy, test_loss, y_true, y_pred_classes, cm

# ---------------------------
# VISUALIZATION
# ---------------------------

def plot_training_history(history, model_name='Model'):
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    plt.plot(history.history.get('accuracy', []), label='Train', linewidth=2)
    plt.plot(history.history.get('val_accuracy', []), label='Val', linewidth=2)
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1,2,2)
    plt.plot(history.history.get('loss', []), label='Train', linewidth=2)
    plt.plot(history.history.get('val_loss', []), label='Val', linewidth=2)
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_history.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, class_names, model_name='Model'):
    """
    Plot confusion matrix - SIMPLIFIED VERSION
    Automatically handles the correct number of classes.
    """
    # The confusion matrix size tells us how many classes are present
    num_classes_in_cm = cm.shape[0]
    
    # Use the first N class names that match the confusion matrix size
    if num_classes_in_cm < len(class_names):
        print(f"\n⚠️  Note: Confusion matrix has {num_classes_in_cm} classes (expected {len(class_names)})")
        display_names = class_names[:num_classes_in_cm]
    else:
        display_names = class_names
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        xticklabels=display_names, 
        yticklabels=display_names, 
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    # Calculate accuracy from confusion matrix
    accuracy = np.trace(cm) / np.sum(cm) * 100
    
    plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {accuracy:.2f}%', 
              fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Confusion matrix saved: {model_name}_confusion_matrix.png")

def plot_sample_images(train_gen, class_names):
    """Show sample images from training set."""
    try:
        batch_images, batch_labels = next(train_gen)
        plt.figure(figsize=(10,10))
        for i in range(min(9, len(batch_images))):
            plt.subplot(3,3,i+1)
            plt.imshow(batch_images[i])
            plt.title(class_names[np.argmax(batch_labels[i])])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=150)
        plt.show()
        print("✓ Sample images saved: sample_images.png\n")
    except Exception as e:
        print(f"⚠️  Could not visualize samples: {e}\n")

# ---------------------------
# MAIN PIPELINE
# ---------------------------

def main():
    print("\n" + "="*80)
    print("LSUN SCENE CLASSIFICATION PIPELINE")
    print("="*80 + "\n")
    
    # Step 1: Validate data
    if not validate_and_diagnose_data():
        print("\n❌ Please fix data issues before training!")
        return
    
    # Step 2: Organize data
    organize_dataset_structure(DATA_ROOT, ORG_DIR, max_images_per_class=MAX_IMAGES_PER_CLASS)
    
    # Step 3: Create data generators
    train_gen, val_gen, test_gen = create_data_generators(str(ORG_DIR), img_size=IMG_SIZE, batch_size=BATCH_SIZE)
        
    # Create SEPARATE generators WITH rescaling (for baseline CNN)
    train_gen_cnn, val_gen_cnn, test_gen_cnn = create_data_generators_for_cnn(str(ORG_DIR), img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())
    
    results = {}
    
    # --- TRAIN MODELS ---
    
    # --- Baseline CNN (use rescaled generators) ---
    print("\n" + "="*80)
    print("[1] BASELINE CNN")
    print("="*80)
    baseline = create_baseline_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes, l2_reg=0.01)
    baseline.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                    loss='categorical_crossentropy', metrics=['accuracy'])
    history_b, t_b = train_model(baseline, train_gen_cnn, val_gen_cnn, epochs=EPOCHS, model_name='baseline_cnn')
    plot_training_history(history_b, 'Baseline_CNN')
    acc, loss, y_true, y_pred, cm = evaluate_model(baseline, test_gen_cnn, 'Baseline_CNN')
    plot_confusion_matrix(cm, class_names, 'Baseline_CNN')  # Removed unique_classes parameter
    results['Baseline CNN'] = {'accuracy': acc, 'loss': loss, 'time': t_b}

    # --- ResNet50 (use NON-rescaled generators) ---
    print("\n" + "="*80)
    print("[2] RESNET50 (Transfer Learning) - FIXED PREPROCESSING")
    print("="*80)
    resnet = create_transfer_learning_model('ResNet50', input_shape=(IMG_SIZE, IMG_SIZE, 3), 
                                           num_classes=num_classes, trainable=False)
    resnet.compile(optimizer=optimizers.Adam(learning_rate=0.0003),  # Lower LR for transfer learning
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history_res, t_res = train_model(resnet, train_gen, val_gen, epochs=EPOCHS, model_name='resnet50_frozen')
    plot_training_history(history_res, 'ResNet50_Frozen')
    acc, loss, y_true, y_pred, cm = evaluate_model(resnet, test_gen, 'ResNet50_Frozen')
    plot_confusion_matrix(cm, class_names, 'ResNet50_Frozen')  # Removed unique_classes parameter
    results['ResNet50 (Frozen)'] = {'accuracy': acc, 'loss': loss, 'time': t_res}

    # --- EfficientNetB0 (use NON-rescaled generators) ---
    print("\n" + "="*80)
    print("[3] EFFICIENTNET-B0 (Transfer Learning) - FIXED PREPROCESSING")
    print("="*80)
    effnet = create_transfer_learning_model('EfficientNetB0', input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                           num_classes=num_classes, trainable=False)
    effnet.compile(optimizer=optimizers.Adam(learning_rate=0.0003),  # Lower LR
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history_eff, t_eff = train_model(effnet, train_gen, val_gen, epochs=EPOCHS, model_name='efficientnet_frozen')
    plot_training_history(history_eff, 'EfficientNet_Frozen')
    acc, loss, y_true, y_pred, cm = evaluate_model(effnet, test_gen, 'EfficientNet_Frozen')
    plot_confusion_matrix(cm, class_names, 'EfficientNet_Frozen')  # Removed unique_classes parameter
    results['EfficientNet (Frozen)'] = {'accuracy': acc, 'loss': loss, 'time': t_eff}
    
    # --- VGG16 (OPTIONAL - use NON-rescaled generators) ---
    # Uncomment if you want to test VGG16
    # print("\n" + "="*80)
    # print("[4] VGG16 (Transfer Learning) - FIXED PREPROCESSING")
    # print("="*80)
    # vgg = create_transfer_learning_model('VGG16', input_shape=(IMG_SIZE, IMG_SIZE, 3),
    #                                     num_classes=num_classes, trainable=False)
    # vgg.compile(optimizer=optimizers.Adam(learning_rate=0.0003),
    #            loss='categorical_crossentropy', metrics=['accuracy'])
    # history_vgg, t_vgg = train_model(vgg, train_gen, val_gen, epochs=EPOCHS, model_name='vgg16_frozen')
    # plot_training_history(history_vgg, 'VGG16_Frozen')
    # acc, loss, y_true, y_pred, cm = evaluate_model(vgg, test_gen, 'VGG16_Frozen')
    # plot_confusion_matrix(cm, class_names, 'VGG16_Frozen')
    # results['VGG16 (Frozen)'] = {'accuracy': acc, 'loss': loss, 'time': t_vgg}
    
    # --- DenseNet121 (OPTIONAL - use NON-rescaled generators) ---
    # Uncomment if you want to test DenseNet121
    # print("\n" + "="*80)
    # print("[5] DENSENET121 (Transfer Learning) - FIXED PREPROCESSING")
    # print("="*80)
    # densenet = create_transfer_learning_model('DenseNet121', input_shape=(IMG_SIZE, IMG_SIZE, 3),
    #                                          num_classes=num_classes, trainable=False)
    # densenet.compile(optimizer=optimizers.Adam(learning_rate=0.0003),
    #                 loss='categorical_crossentropy', metrics=['accuracy'])
    # history_dn, t_dn = train_model(densenet, train_gen, val_gen, epochs=EPOCHS, model_name='densenet121_frozen')
    # plot_training_history(history_dn, 'DenseNet121_Frozen')
    # acc, loss, y_true, y_pred, cm = evaluate_model(densenet, test_gen, 'DenseNet121_Frozen')
    # plot_confusion_matrix(cm, class_names, 'DenseNet121_Frozen')
    # results['DenseNet121 (Frozen)'] = {'accuracy': acc, 'loss': loss, 'time': t_dn}

    # Summary comparison
    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\n{'Model':<30} {'Accuracy':<12} {'Loss':<12} {'Time (min)':<12}")
    print("-" * 70)
    for name, info in results.items():
        print(f"{name:<30} {info['accuracy']:<12.4f} {info['loss']:<12.4f} {info['time']/60:<12.1f}")
    print("="*80)

    print("\n✓ Pipeline finished successfully!")
    return results

if __name__ == '__main__':
    main()
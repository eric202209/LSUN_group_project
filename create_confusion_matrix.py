# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 09:38:25 2025

@author: Eric
"""

"""
Script to load a saved model and generate confusion matrix without retraining.
Works with any saved .keras or .h5 model file.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

# ---------------------------
# CONFIGURATION
# ---------------------------
MODEL_PATH = 'resnet50_frozen_best.keras'  # Change to your model path
DATA_DIR = Path('E:/lsun_tra_val')  # Your organized data directory
IMG_SIZE = 224  # MUST MATCH training size! Error showed: expected (224, 224, 3)
BATCH_SIZE = 32

# Your class names (in the same order as during training)
CLASS_NAMES = ['bedroom', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room']

# ---------------------------
# LOAD MODEL
# ---------------------------
print("="*80)
print("LOADING SAVED MODEL")
print("="*80)
print(f"\nLooking for model: {MODEL_PATH}")

# Check if file exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    print(f"\nSearched in: {os.path.abspath(MODEL_PATH)}")
    print("\nPlease update MODEL_PATH at the top of this script with the correct path.")
    print("Use one of the paths shown in 'Found model files' above.\n")
    import sys
    sys.exit(1)

print(f"✓ File found: {MODEL_PATH}")

try:
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
    print(f"\nModel summary:")
    model.summary()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nMake sure:")
    print(f"  1. File is a valid Keras model (.keras or .h5)")
    print(f"  2. File is not corrupted")
    import sys
    sys.exit(1)

# ---------------------------
# CREATE TEST DATA GENERATOR
# ---------------------------
print("\n" + "="*80)
print("CREATING TEST DATA GENERATOR")
print("="*80)

# IMPORTANT: Use NO rescaling for transfer learning models
# (They have preprocessing built-in)
test_datagen = ImageDataGenerator()  # No rescaling, no augmentation

test_dir = DATA_DIR / 'test'

if not test_dir.exists():
    print(f"❌ Error: Test directory not found: {test_dir}")
    import sys
    sys.exit(1)

test_generator = test_datagen.flow_from_directory(
    str(test_dir),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # IMPORTANT: Don't shuffle for evaluation
)

print(f"\n✓ Test generator created:")
print(f"  Test samples: {test_generator.samples:,}")
print(f"  Classes found: {list(test_generator.class_indices.keys())}")
print(f"  Batch size: {BATCH_SIZE}")

# ---------------------------
# EVALUATE MODEL
# ---------------------------
print("\n" + "="*80)
print("EVALUATING MODEL ON TEST SET")
print("="*80)

# Get predictions
print("\nGenerating predictions...")
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Calculate accuracy
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"\n✓ Evaluation complete:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# ---------------------------
# CLASSIFICATION REPORT
# ---------------------------
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80 + "\n")

# Handle case where some classes might be missing from test set
unique_classes = sorted(np.unique(y_true))
if len(unique_classes) < len(CLASS_NAMES):
    print(f"⚠️  Warning: Only {len(unique_classes)} out of {len(CLASS_NAMES)} classes in test set")
    target_names = [CLASS_NAMES[i] for i in unique_classes]
    labels = unique_classes
else:
    target_names = CLASS_NAMES
    labels = None

print(classification_report(
    y_true, 
    y_pred_classes, 
    labels=labels,
    target_names=target_names, 
    digits=4,
    zero_division=0
))

# ---------------------------
# CONFUSION MATRIX
# ---------------------------
print("\n" + "="*80)
print("GENERATING CONFUSION MATRIX")
print("="*80 + "\n")

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred_classes, labels=labels)

print("Confusion Matrix:")
print(cm)

# ---------------------------
# PLOT CONFUSION MATRIX
# ---------------------------
def plot_confusion_matrix(cm, class_names, model_name='Model', save_path=None):
    """
    Create a beautiful confusion matrix visualization.
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True,  # Show numbers
        fmt='d',  # Integer format
        cmap='Blues',  # Color scheme
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'},
        square=True,  # Make cells square
        linewidths=0.5,  # Add grid lines
        linecolor='gray'
    )
    
    plt.title(f'{model_name} - Confusion Matrix\nAccuracy: {np.trace(cm)/np.sum(cm)*100:.2f}%', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = f'{model_name}_confusion_matrix.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {save_path}")
    
    plt.show()

# Plot the confusion matrix
model_name = Path(MODEL_PATH).stem  # Extract model name from path
plot_confusion_matrix(cm, target_names, model_name=model_name)

# ---------------------------
# PER-CLASS ACCURACY
# ---------------------------
print("\n" + "="*80)
print("PER-CLASS ACCURACY")
print("="*80 + "\n")

per_class_acc = cm.diagonal() / cm.sum(axis=1)

print(f"{'Class':<20} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
print("-" * 55)
for i, class_name in enumerate(target_names):
    correct = cm[i, i]
    total = cm[i].sum()
    accuracy = per_class_acc[i]
    print(f"{class_name:<20} {accuracy:>8.2%}   {correct:>8}   {total:>8}")

print(f"\n{'Overall Accuracy':<20} {test_accuracy:>8.2%}")

# ---------------------------
# TOP CONFUSIONS (Most common mistakes)
# ---------------------------
print("\n" + "="*80)
print("TOP CONFUSIONS (Most Common Mistakes)")
print("="*80 + "\n")

# Find off-diagonal elements (mistakes)
mistakes = []
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if i != j and cm[i, j] > 0:  # Not diagonal and has mistakes
            mistakes.append((cm[i, j], target_names[i], target_names[j]))

# Sort by count (most mistakes first)
mistakes.sort(reverse=True)

print(f"{'Count':<8} {'True Class':<20} {'Predicted As':<20}")
print("-" * 55)
for count, true_class, pred_class in mistakes[:10]:  # Top 10 mistakes
    print(f"{count:<8} {true_class:<20} {pred_class:<20}")

if not mistakes:
    print("🎉 No mistakes! Perfect classification!")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print("="*80)
print(f"\nSaved files:")
print(f"  - {model_name}_confusion_matrix.png")
print("\nYou can now use this confusion matrix in your report!")
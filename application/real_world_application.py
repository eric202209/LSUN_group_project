# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:23:13 2025

@author: Eric
"""

"""
APPLICATIONS FOR SCENE CLASSIFICATION MODELS

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
from PIL import Image

from tensorflow import keras
# Import necessary preprocessing functions
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# ==============================================================================
# 1. CONFIGURATION & SETUP
# ==============================================================================
# Define all trained models and their specific preprocessing functions
MODEL_CONFIGS = {
    'baseline_cnn': {
        'path': 'baseline_cnn_best.keras',
        'preprocess_fn': lambda x: x / 255.0 # Assuming simple CNN uses 0-1 scaling
    },
    'efficientnet_frozen': {
        'path': 'efficientnet_frozen_best.keras',
        'preprocess_fn': efficientnet_preprocess
    },
    'resnet50_frozen': {
        'path': 'resnet50_frozen_best.keras',
        'preprocess_fn': resnet_preprocess
    }
}

IMG_SIZE = 224
CLASS_NAMES = ['bedroom', 'classroom', 'conference_room', 'dining_room', 'kitchen', 'living_room']

# --- Set the model for deployment ---
DEPLOYMENT_MODEL_KEY = 'efficientnet_frozen' 
CONFIDENCE_THRESHOLD = 0.70 # Threshold for flagging manual review


# ==============================================================================
# 2. BATCH RENAME UTILITY
# ==============================================================================
def batch_rename_input_photos(input_folder='test_photos', base_name='ListingPhoto'):
    """
    Renames all image files in the input_folder sequentially (e.g., ListingPhoto_001.jpg) 
    using the os.rename method for better cross-platform compatibility.
    """
    print("="*60)
    print(f"🔄 Starting Batch Rename in: {input_folder}")
    print("="*60)
    
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ Error: Input folder not found: {input_folder}")
        return

    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_files = []
    for ext in image_extensions:
        # Glob for both lowercase and uppercase extensions
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))

    if not image_files:
        print(f"❌ No images found in {input_folder} to rename.")
        return

    # Sort files to ensure predictable sequential numbering
    image_files.sort()
    
    renamed_count = 0
    for i, old_path in enumerate(image_files, 1):
        extension = old_path.suffix.lower()
        new_name = f"{base_name}_{i:03}{extension}"
        new_path = input_path / new_name
        
        try:
            # Use os.rename with absolute paths for robustness
            os.rename(str(old_path.resolve()), str(new_path.resolve()))
            renamed_count += 1
        except Exception as e:
            print(f"  ⚠️ Failed to rename {old_path.name}: {e}")

    print(f"\n✅ Successfully renamed {renamed_count} photos (e.g., {base_name}_001.jpg).")


# ==============================================================================
# 3. ROOM CLASSIFIER APPLICATION CLASS
# ==============================================================================
class RoomOrganizerApp:
    def __init__(self, model_key=DEPLOYMENT_MODEL_KEY):
        """Initializes the application, loading the chosen deployment model."""
        self.model_key = model_key
        config = MODEL_CONFIGS[model_key]
        self.model_path = config['path']
        self.preprocess_fn = config['preprocess_fn']
        self.model = self._load_model()
        
    def _load_model(self):
        """Helper to load the Keras model."""
        print("="*60)
        print(f"Loading Model: **{self.model_key.upper()}** from {self.model_path}...")
        if not Path(self.model_path).exists():
            print(f"❌ Error: Model file not found at '{self.model_path}'")
            return None
        try:
            model = keras.models.load_model(self.model_path)
            print("✓ Model loaded successfully!")
            print("="*60 + "\n")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def _predict_room_type(self, image_path):
        """Core prediction logic used by all functions."""
        if self.model is None:
            raise ValueError("Deployment Model not loaded.")

        # Load, resize, convert to array, and preprocess
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = self.preprocess_fn(img_array)
        
        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = predictions[predicted_idx]
        
        all_probs = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
        return predicted_class, confidence, all_probs

    # --- Main Application Feature ---
    def organize_photos_and_report(self, input_folder='test_photos', output_folder='organized_listings_demo', log_file='prediction_log.csv'):
        """
        Automatically classifies and sorts a batch of images, then generates a summary report.
        """
        if self.model is None: return

        print("\n" + "="*80)
        print(f"📸 REAL ESTATE PHOTO AUTO-ORGANIZER (Model: {self.model_key.upper()})")
        print("="*80)
        
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists():
            print(f"❌ Input folder not found: {input_folder}. Please create it and add photos.")
            return
        
        # Setup output folders for each class
        for room_type in CLASS_NAMES:
            (output_path / room_type).mkdir(parents=True, exist_ok=True)
            
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = [p for ext in image_extensions for p in input_path.glob(f'*{ext}')]
        
        if len(image_files) == 0:
            print(f"❌ No images found in {input_folder}")
            return

        print(f"Found {len(image_files)} photos to process...")
        log_data = []

        # Process and log
        for i, img_file in enumerate(image_files, 1):
            try:
                room_type, confidence, all_probs = self._predict_room_type(img_file)
                
                # Copy file to its predicted folder
                dest = output_path / room_type / img_file.name
                shutil.copy2(img_file, dest)
                
                # Create log entry
                log_entry = {
                    'filename': img_file.name,
                    'predicted_room': room_type,
                    'confidence': confidence,
                    'status': 'Sorted' if confidence >= CONFIDENCE_THRESHOLD else 'Review_Low_Confidence'
                }
                # Add all probabilities to the log for advanced analysis
                log_entry.update(all_probs)
                log_data.append(log_entry)
                
                print(f"  {i}/{len(image_files)}: {img_file.name:30} → {room_type.upper():15} ({confidence:.1%})")

            except Exception as e:
                log_data.append({'filename': img_file.name, 'predicted_room': 'Error', 'confidence': 0.0, 'status': f"Failed: {e}"})

        # Generate and display reports
        if log_data:
            self._generate_report(log_data, output_folder, log_file)
        
        print("\n" + "="*80)
        print(f"✅ ORGANIZATION COMPLETE! Output folder: {output_path.absolute()}")
        print("==================================================================")

    # --- Reporting and Plotting ---
    def _generate_report(self, log_data, output_folder, log_file):
        """Creates a CSV log and all presentation reports/plots."""
        
        df = pd.DataFrame(log_data)
        log_path = Path(output_folder) / log_file
        df.to_csv(log_path, index=False)
        print(f"\n📝 Detailed prediction log saved to: **{log_path}**")

        # 1. Distribution Plot
        self.plot_distribution(df)
        
        # 2. Confidence Histogram (New)
        self.plot_confidence_histogram(df)

        # 3. Low Confidence Breakdown (New)
        self.plot_low_confidence_breakdown(df)
        
        # Display flagged photos summary
        low_conf_df = df[df['status'] == 'Review_Low_Confidence'].sort_values(by='confidence')
        if not low_conf_df.empty:
            print(f"\n⚠️ {len(low_conf_df)} Photos Flagged for Manual Review (<{CONFIDENCE_THRESHOLD:.0%} Confidence):")
            for _, row in low_conf_df.head(5).iterrows():
                print(f"   - {row['filename']:30} → {row['predicted_room']:15} ({row['confidence']:.1%})")
            if len(low_conf_df) > 5:
                print(f"   ... and {len(low_conf_df)-5} more. Check the CSV log for details.")


    def plot_distribution(self, df):
        """Generates the primary room type distribution plot."""
        plt.figure(figsize=(10, 6))
        room_counts = df['predicted_room'].value_counts()
        
        if 'Error' in room_counts: room_counts = room_counts.drop('Error')

        if not room_counts.empty:
            room_counts.sort_values(ascending=True).plot(kind='barh', color='skyblue')
            plt.title('1. Distribution of Organized Photos by Room Type')
            plt.xlabel('Number of Photos')
            plt.ylabel('Room Type')
            plt.tight_layout()
            plt.show()
            
    def plot_confidence_histogram(self, df):
        """Generates a histogram of prediction confidence scores to show reliability."""
        plt.figure(figsize=(9, 5))
        valid_confidences = df[df['status'] != 'Failed']['confidence']
        
        plt.hist(valid_confidences, bins=np.linspace(0, 1, 21), edgecolor='black', color='teal', alpha=0.7)
        
        plt.title(f'2. Prediction Confidence Distribution (Model: {self.model_key.upper()})')
        plt.xlabel('Prediction Confidence Score (0.0 to 1.0)')
        plt.ylabel('Number of Photos')
        plt.xticks(np.linspace(0, 1, 11))
        plt.grid(axis='y', alpha=0.5)
        
        # Add a vertical line at the review threshold
        plt.axvline(x=CONFIDENCE_THRESHOLD, color='red', linestyle='--', label=f'Manual Review Threshold ({CONFIDENCE_THRESHOLD:.0%})')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_low_confidence_breakdown(self, df):
        """Plots a horizontal bar chart showing the breakdown of photos flagged for manual review."""
        low_conf_df = df[df['status'] == 'Review_Low_Confidence']
        
        if low_conf_df.empty:
            return

        plt.figure(figsize=(9, 5))
        low_conf_counts = low_conf_df['predicted_room'].value_counts()
        
        low_conf_counts.sort_values(ascending=True).plot(kind='barh', color='darkorange', alpha=0.8)
        
        plt.title(f'3. Breakdown of Photos Flagged for Manual Review ({len(low_conf_df)} Total)')
        plt.xlabel('Number of Photos Requiring Review')
        plt.ylabel('Predicted Room Type')
        plt.tight_layout()
        plt.show()


# ==============================================================================
# 4. MODEL COMPARISON TOOL (FOR PRESENTATION)
# ==============================================================================
def compare_all_models_on_image(image_path='test_room.jpg'):
    """Runs a single image through all models for performance comparison."""
    
    def static_predict(model_name, image_path):
        config = MODEL_CONFIGS[model_name]
        temp_model = keras.models.load_model(config['path'])
        preprocess_fn = config['preprocess_fn']
        
        img = Image.open(image_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_fn(img_array)
        
        predictions = temp_model.predict(img_array, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        room_type = CLASS_NAMES[predicted_idx]
        return room_type, confidence

    print("\n" + "="*80)
    print("📈 MODEL PERFORMANCE COMPARISON: Single Image Test")
    print("="*80)
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}. Cannot perform comparison.")
        return

    results = {}
    max_confidence = -1
    
    for model_name in MODEL_CONFIGS.keys():
        try:
            room_type, confidence = static_predict(model_name, image_path)
            results[model_name] = {'prediction': room_type, 'confidence': confidence}
            max_confidence = max(max_confidence, confidence)
        except Exception as e:
            results[model_name] = {'error': f"Load/Prediction Error: {e}"}
    
    # Print results table
    print(f"📷 Analyzing image: {Path(image_path).name}\n")
    print(f"{'Model':<20} | {'Prediction':<15} | {'Confidence':<25}")
    print("-" * 65)
    
    for model_name, data in results.items():
        if 'error' in data:
             print(f"{model_name:<20} | {data['error']}")
        else:
            conf_str = f"{data['confidence']:.2%}"
            if data['confidence'] == max_confidence:
                conf_str = f"**{conf_str}** (HIGHEST)"
            
            print(f"{model_name:<20} | {data['prediction']:<15} | {conf_str}")
            
    print("\n💡 Presentation Value: Justifies using the most confident model for deployment.")


# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    
    # --- DEMO SETUP ---
    INPUT_FOLDER = 'test_photos'
    OUTPUT_FOLDER = 'organized_listings_demo'
    TEST_IMAGE = 'test_image.jpg'
    
    # 1. OPTIONAL: Batch Rename (Run this if your filenames are messy)
    batch_rename_input_photos(input_folder=INPUT_FOLDER) 
    
    # 2. Initialize the App with the robust model
    app = RoomOrganizerApp(model_key=DEPLOYMENT_MODEL_KEY) 

    if app.model:
        # 3. Run the Batch Organizer, Sorting, Logging, and Plotting
        print("\n\n--- RUNNING BATCH PHOTO ORGANIZER AND REPORTING ---")
        app.organize_photos_and_report(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER)
        
        # 4. Run the Model Comparison for Presentation
        print("\n\n--- RUNNING MODEL COMPARISON DEMO ---")
        compare_all_models_on_image(image_path=TEST_IMAGE)

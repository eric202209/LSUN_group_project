# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 16:54:32 2025

@author: Eric
"""

"""
Simple script to convert all .webp images to .jpg in living_room folder.
No backup, fast conversion for 30,000 images.
"""

from pathlib import Path
from PIL import Image
import time

# ---------------------------
# CONFIGURATION
# ---------------------------
SOURCE_DIR = Path('E:/lsun_raw/living_room')

print("="*80)
print("CONVERT WEBP TO JPG - LIVING ROOM")
print("="*80)

# Check if directory exists
if not SOURCE_DIR.exists():
    print(f"\n❌ Directory not found: {SOURCE_DIR}")
    exit()

# Find all .webp files
print(f"\nScanning directory: {SOURCE_DIR}")
webp_files = list(SOURCE_DIR.glob('*.webp'))

print(f"Found {len(webp_files):,} .webp files to convert")

if len(webp_files) == 0:
    print("\n✅ No .webp files found! Already converted or wrong directory?")
    exit()

# Confirm before proceeding
print("\n⚠️  This will:")
print(f"   1. Convert {len(webp_files):,} .webp files to .jpg")
print(f"   2. Delete original .webp files (NO BACKUP)")
print(f"   3. Save as JPG with 95% quality")

response = input("\nProceed? (yes/no): ").lower()
if response != 'yes':
    print("❌ Cancelled by user")
    exit()

# ---------------------------
# CONVERT FILES
# ---------------------------
print("\n" + "="*80)
print("CONVERTING FILES")
print("="*80 + "\n")

converted = 0
failed = 0
start_time = time.time()

for i, webp_file in enumerate(webp_files, 1):
    try:
        # Create .jpg filename
        jpg_file = webp_file.with_suffix('.jpg')
        
        # Skip if .jpg already exists
        if jpg_file.exists():
            webp_file.unlink()  # Delete .webp since .jpg exists
            converted += 1
            continue
        
        # Open webp image
        img = Image.open(webp_file)
        
        # Convert to RGB if needed (webp can have alpha channel)
        if img.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparency
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            # Paste with alpha channel as mask
            if img.mode == 'RGBA':
                rgb_img.paste(img, mask=img.split()[3])
            else:
                rgb_img.paste(img)
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as JPG with high quality
        img.save(jpg_file, 'JPEG', quality=95, optimize=True)
        
        # Close image to free memory
        img.close()
        
        # Delete original .webp file
        webp_file.unlink()
        
        converted += 1
        
        # Progress update every 500 images
        if i % 500 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(webp_files) - i) / rate if rate > 0 else 0
            print(f"  Progress: {i:,}/{len(webp_files):,} ({i/len(webp_files)*100:.1f}%) - "
                  f"Rate: {rate:.1f} img/s - ETA: {remaining/60:.1f} min")
    
    except Exception as e:
        print(f"  ❌ Failed: {webp_file.name} - {e}")
        failed += 1

# ---------------------------
# SUMMARY
# ---------------------------
elapsed_time = time.time() - start_time

print("\n" + "="*80)
print("CONVERSION COMPLETE")
print("="*80)
print(f"\n✅ Converted: {converted:,} files")
if failed > 0:
    print(f"❌ Failed: {failed:,} files")
print(f"\n⏱️  Time taken: {elapsed_time/60:.1f} minutes ({elapsed_time:.1f} seconds)")
print(f"📊 Average speed: {converted/elapsed_time:.1f} images/second")

# Verify conversion
remaining_webp = list(SOURCE_DIR.glob('*.webp'))
jpg_files = list(SOURCE_DIR.glob('*.jpg'))

print(f"\n📁 Final counts:")
print(f"   .webp files remaining: {len(remaining_webp):,}")
print(f"   .jpg files: {len(jpg_files):,}")

if len(remaining_webp) == 0:
    print("\n🎉 SUCCESS! All .webp files converted to .jpg")
    print("\nNext steps:")
    print("   1. Delete E:\\lsun_tra_val (if it exists)")
    print("   2. Re-run your training script to reorganize data")
    print("   3. All 6 classes should now be detected properly!")
else:
    print(f"\n⚠️  Warning: {len(remaining_webp):,} .webp files still remain")
    print("   Some files may have failed to convert")

print("\n" + "="*80)
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def prepare_dataset():
    print("🔄 Preparing dataset for YOLO...")
    
    # Animal classes
    classes = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 
               'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
    
    class_map = {
        'cane': 'dog', 'cavallo': 'horse', 'elefante': 'elephant',
        'farfalla': 'butterfly', 'gallina': 'chicken', 'gatto': 'cat',
        'mucca': 'cow', 'pecora': 'sheep', 'ragno': 'spider',
        'scoiattolo': 'squirrel'
    }
    
    # Get all images
    raw_path = Path('data/raw/raw-img')
    all_images = []
    
    for class_name in classes:
        class_path = raw_path / class_name
        if class_path.exists():
            images = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            all_images.extend([(img, class_name) for img in images])
            print(f"  Found {len(images)} images of {class_map[class_name]}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(all_images)
    
    train_size = int(len(all_images) * 0.8)
    val_size = int(len(all_images) * 0.1)
    
    splits = {
        'train': all_images[:train_size],
        'val': all_images[train_size:train_size + val_size],
        'test': all_images[train_size + val_size:]
    }
    
    # Process each split
    for split_name, split_data in splits.items():
        print(f"\n📦 Processing {split_name} ({len(split_data)} images)...")
        
        img_dir = Path(f'data/{split_name}/images')
        label_dir = Path(f'data/{split_name}/labels')
        
        for img_path, class_name in tqdm(split_data):
            # Copy image
            new_name = f"{class_map[class_name]}_{img_path.name}"
            shutil.copy2(img_path, img_dir / new_name)
            
            # Create label (full image bounding box for now)
            class_idx = list(class_map.values()).index(class_map[class_name])
            label_content = f"{class_idx} 0.5 0.5 1.0 1.0\n"
            
            label_file = label_dir / f"{Path(new_name).stem}.txt"
            label_file.write_text(label_content)
    
    print("\n✅ Dataset preparation complete!")
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")

if __name__ == "__main__":
    prepare_dataset()
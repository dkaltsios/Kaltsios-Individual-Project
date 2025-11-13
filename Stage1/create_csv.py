import os
import pandas as pd

def create_dataset_csv(malign_dir, benign_dir, output_file='skin_lesions.csv'):
    # Get all image files from both directories
    malign_images = [f for f in os.listdir(malign_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    benign_images = [f for f in os.listdir(benign_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create dataframe
    data = []
    for img in malign_images:
        data.append({'image_path': img, 'is_malign': 1})
    for img in benign_images:
        data.append({'image_path': img, 'is_malign': 0})
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(df)} images")

if __name__ == '__main__':
    create_dataset_csv('MalignImages', 'NEVSEKimages')
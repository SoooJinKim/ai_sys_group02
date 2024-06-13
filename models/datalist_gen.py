import os
import random

def get_image_pairs(original_dir, overlay_dir):
    image_pairs = []
    for file in os.listdir(original_dir):
        if file.endswith('.png'):
            base_name = file
            overlay_name = f"high_masked_processed_{base_name}"
            original_path = os.path.join(os.path.abspath(original_dir), file)
            overlay_path = os.path.join(os.path.abspath(overlay_dir), overlay_name)
            if os.path.exists(overlay_path):
                image_pairs.append((original_path, overlay_path))
    return image_pairs

def split_dataset(image_pairs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    random.shuffle(image_pairs)
    total = len(image_pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return image_pairs[:train_end], image_pairs[train_end:val_end], image_pairs[val_end:]

def save_paths_to_txt(image_pairs, original_file_path, overlay_file_path):
    with open(original_file_path, 'w') as f_org, open(overlay_file_path, 'w') as f_ovl:
        for original_path, overlay_path in image_pairs:
            f_org.write(f"{original_path}\n")
            f_ovl.write(f"{overlay_path}\n")

def main():
    original_dir = 'C:\\Users\\Serin Kim\\workspace\\AISYS\\DMFN-master\\AISYS_data\\original'
    overlay_dir = 'C:\\Users\\Serin Kim\\workspace\\AISYS\\DMFN-master\\AISYS_data\\overlay'
    output_dir = 'C:\\Users\\Serin Kim\\workspace\\AISYS\\DMFN-master\\datasets\\aisys'
    os.makedirs(output_dir, exist_ok=True)

    image_pairs = get_image_pairs(original_dir, overlay_dir)
    train_pairs, val_pairs, test_pairs = split_dataset(image_pairs)

    save_paths_to_txt(train_pairs, os.path.join(output_dir, 'train_original_list.txt'), os.path.join(output_dir, 'train_overlay_list.txt'))
    save_paths_to_txt(val_pairs, os.path.join(output_dir, 'val_original_list.txt'), os.path.join(output_dir, 'val_overlay_list.txt'))
    save_paths_to_txt(test_pairs, os.path.join(output_dir, 'test_original_list.txt'), os.path.join(output_dir, 'test_overlay_list.txt'))

if __name__ == '__main__':
    main()

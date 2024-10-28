import os
import shutil
import json

def load_image_filenames(json_file):
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    

    return [(img['id'], img['file_name']) for img in data['images']]


def load_image_filenames(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    return [(img['id'], f"{str(img['id']).zfill(4)}.jpg") for img in data['images']]


def copy_images(image_filenames, src_folder, dest_folder):
    
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for image_id, filename in image_filenames:
        src_image_path = os.path.join(src_folder, filename)
        dest_image_path = os.path.join(dest_folder, filename)

        if os.path.exists(src_image_path):
            shutil.copy(src_image_path, dest_image_path)
            print(f"Copied {filename} to {dest_folder}")
        else:
            print(f"Image {filename} not found in {src_folder}")

def main():
    root = '/data/ephemeral/home'
    src_folder = os.path.join(root, 'dataset/train')
    val_json_path = os.path.join(root, 'dataset/val_fold_3.json')
    train_json_path = os.path.join(root, 'dataset/train_fold_3.json')
    
    val_images_folder = os.path.join(root, 'dataset/val_3')
    train_images_folder = os.path.join(root, 'dataset/train_3')

    
    val_image_filenames = load_image_filenames(val_json_path)
    train_image_filenames = load_image_filenames(train_json_path)

    # 이미지 파일 복사
    copy_images(val_image_filenames, src_folder, val_images_folder)
    copy_images(train_image_filenames, src_folder, train_images_folder)

if __name__ == "__main__":
    main()

import os
import argparse
import pickle
import cv2
import numpy as np

CIFAR_TREE = "train/cifar-100-python"

def get_data(base_dir, split):
    bin_path = os.path.join(base_dir, CIFAR_TREE, split)
    with open(bin_path, "rb") as bin_file:
        all_data = pickle.load(bin_file, encoding="latin1")
    images = np.vstack(all_data["data"]).reshape(-1, 3, 32, 32)
    images = images.transpose((0, 2, 3, 1))
    labels = all_data["fine_labels"]
    return images, labels

def save_files(root_dir, np_images, labels):
    image_path = os.path.join(root_dir, "images")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    file_list_path = os.path.join(root_dir, "file_list.txt")

    with open(file_list_path, "w") as file_list:
        for i in range(np_images.shape[0]):
            cv2.imwrite(os.path.join(image_path, f"{i}.jpg"), np_images[i])
            file_list.write(f"images/{i}.jpg {labels[i]}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="../bash_files/pretrain/cifar/datasets/cifar100")
    parser.add_argument("--output_root", default="../data_files/cifar100")
    args = parser.parse_args()


    for split in ["train", "test"]:
        np_images, labels = get_data(args.base_dir, split)
        save_files(os.path.join(args.output_root, split), np_images, labels)

if __name__ == "__main__":
    main()

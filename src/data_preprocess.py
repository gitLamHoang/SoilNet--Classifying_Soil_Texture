import cv2
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader
def preprocess_image(img_path, size=256, crop_border = 10):
   img = cv2.imread(img_path)
   if img is None:
      return None
   h, w = img.shape[:2]
   if h<= 2*crop_border or w <= 2*crop_border:
      return None
   img = img[crop_border:h-crop_border, crop_border:w-crop_border]
   img = cv2.resize(img, (size, size))
   return img

def read_img(input_path, label_array):
    images, labels = [], []
    for index, label in enumerate(label_array):
        label_path = os.path.join(input_path, label)
        imgs = os.listdir(label_path)

        class_count = 0
        for j, img_file in enumerate(imgs):
            img_path = os.path.join(label_path, img_file)
            img_processed = preprocess_image(img_path, size=256)
            if img_processed is None:
                continue
            images.append(img_processed)
            labels.append(index)
            class_count += 1

        print(f"Class {label}: {class_count} after processed")

    return images, labels

def convert_to_tensor(input_path, label_array):
    images_list, type_list_not_fixed = read_img(input_path, label_array)
    type_list = []
    for i in range (len(type_list_not_fixed)):
        if type_list_not_fixed[i] == 0:
            type_list.append([1.0, 0, 0])
        elif type_list_not_fixed[i] == 1:
            type_list_not_fixed.append([0, 1.0, 0])
        else:
           type_list_not_fixed.append([0, 0, 1.0])
        type_list = np.array(type_list)
    images_array = np.array(images_list, dtype=np.float32)/255.0
    images_array = np.transpose(images_array, (0,3,1,2))
    images_tensor = torch.tensor(images_array, dtype = torch.float32)
    type_array = np.array(type_list, dtype = np.int64)
    type_tensor = torch.tensor(type_array, dtype = torch.long)
    return images_tensor, type_tensor


def train_and_test_loader(input_path, label_array, train_ratio):
    images_tensor, type_tensor = convert_to_tensor(input_path, label_array)
    dataset = TensorDataset(images_tensor, type_tensor)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    state = torch.Generator().manual_seed(30)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator = state)
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = True)
    return train_loader, test_loader

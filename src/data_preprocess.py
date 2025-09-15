import cv2
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split, DataLoader

def read_img(input, label_array):
  images = []
  labels = []
  for index, label in enumerate(label_array): #stt -> gia tri
    label_path = os.path.join(input, label)
    for j, img in enumerate(os.listdir(label_path)):
      img_path = os.path.join(label_path, img)
      img = cv2.imread(img_path)
      if img is None:
        continue
      img_resized = cv2.resize(img, (256,256))
      images.append(img_resized)
      labels.append(index)
      print('Label ' + str(label) + ' images no.' + str(j))
    images = np.array(images)
    images = images/images.max()
  return images, labels


def convert_to_tensor(input, label_array):
    images_list, type_list_not_fixed = read_img(input, label_array)
    type_list = []
    for i in range (len(type_list_not_fixed)):
        if type_list_not_fixed[i] == 0:
            type_list.append([1.0, 0, 0])
        elif type_list_not_fixed[i] == 1:
            type_list_not_fixed.append([0, 1.0, 0])
        else:
           type_list_not_fixed.append([0, 0, 1.0])
        type_list = np.array(type_list)
    
    images_tensor = torch.tensor(images_list, dtype = torch.float32)
    type_tensor = torch.tensor(type_list, dtype = torch.float32)
    return images_tensor, type_tensor


def train_and_test_loader(input, label_array, train_ratio):
    images_tensor, type_tensor = convert_to_tensor(input, label_array)
    dataset = TensorDataset(images_tensor, type_tensor)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    state = torch.Generator().manual_seed(30)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator = state)
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = True)
    return train_loader, test_loader

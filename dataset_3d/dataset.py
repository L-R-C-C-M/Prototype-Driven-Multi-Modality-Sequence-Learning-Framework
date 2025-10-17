# input: (batch_size, 64, 1, 224, 224)

import os
import pdb
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

def load_images_to_array(full_path,size=(224,224)):
    # 获取文件夹中的所有图片文件名
    image_files = [f for f in os.listdir(full_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort()

    # 初始化一个空的 NumPy 数组，形状为 (64,128, 128)
    image_array = np.zeros((len(image_files),size[0], size[1]), dtype=np.uint8)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(full_path, image_file)
        # 以灰度模式打开图片
        image = Image.open(image_path).convert('L')
        # 确保图片大小为 128x128
        if image.size != size:
            image = image.resize(size)
        # 将图片数据转换为 NumPy 数组，并赋值到最终数组中
        image_array[idx, :, :] = np.array(image)
    return image_array

def load_images_to_array_RGB(full_path, size=(224, 224)):
    # 获取文件夹中的所有图片文件名
    image_files = [f for f in os.listdir(full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort()

    # 初始化一个空的 NumPy 数组，形状为 (N, H, W, 3)
    num_images = len(image_files)
    image_array = np.zeros((num_images, size[0], size[1], 3), dtype=np.uint8)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(full_path, image_file)
        # 使用 RGB 模式打开图片
        image = Image.open(image_path).convert('RGB')
        # 调整大小
        if image.size != size:
            image = image.resize(size)
        # 转换为 numpy 数组并放入 array
        image_array[idx] = np.array(image)

    return image_array
# EGFR基因的：
egfr_train_patient_list = ['AMC-003', 'AMC-010', 'AMC-012', 'AMC-013', 'AMC-022', 'AMC-023', 'AMC-027', 'AMC-030',
                      'AMC-032', 'AMC-036', 'AMC-038', 'AMC-041', 'AMC-042', 'AMC-045', 'R01-003', 'R01-026',
                      'R01-037', 'R01-047', 'R01-075', 'R01-076', 'R01-080', 'R01-087', 'R01-088', 'R01-099',
                      'AMC-001', 'AMC-004', 'AMC-009', 'AMC-011', 'AMC-014', 'AMC-015', 'AMC-016', 'AMC-017', 'AMC-018',
                      'AMC-020',
                      'AMC-021', 'AMC-024', 'AMC-025', 'AMC-026', 'AMC-037', 'AMC-039', 'AMC-040', 'AMC-046', 'AMC-049',
                      'R01-004',
                      'R01-006', 'R01-007', 'R01-008', 'R01-009', 'R01-010', 'R01-012', 'R01-015', 'R01-016', 'R01-018',
                      'R01-019',
                      'R01-023', 'R01-027', 'R01-028', 'R01-029', 'R01-031', 'R01-032', 'R01-034', 'R01-035', 'R01-036',
                      'R01-040',
                      'R01-041', 'R01-045', 'R01-046', 'R01-048', 'R01-050', 'R01-051', 'R01-054', 'R01-055', 'R01-057',
                      'R01-059',
                      'R01-060', 'R01-062', 'R01-063', 'R01-065', 'R01-067', 'R01-068', 'R01-069', 'R01-070', 'R01-071',
                      'R01-072',
                      'R01-073', 'R01-078', 'R01-081', 'R01-082', 'R01-083', 'R01-084', 'R01-086', 'R01-090', 'R01-091',
                      'R01-094',
                      'R01-095', 'R01-096', 'R01-097', ]
egfr_val_patient_list = ['R01-104', 'R01-127', 'R01-131', 'R01-132', 'R01-136', 'R01-141', 'R01-142', 'R01-146',
                    'R01-147', 'R01-162',
                    'R01-100', 'R01-103', 'R01-105', 'R01-106', 'R01-110', 'R01-112', 'R01-113',
                    'R01-115', 'R01-116', 'R01-117', 'R01-118', 'R01-119', 'R01-122', 'R01-125', 'R01-126', 'R01-128',
                    'R01-129',
                    'R01-130', 'R01-133', 'R01-134', 'R01-135', 'R01-139', 'R01-140', 'R01-144', 'R01-145', 'R01-148',
                    'R01-149',
                    'R01-151', 'R01-152', 'R01-161']
kras_train_patient_list=[]
kras_val_patient_list=[]

class NSCLC_CT_Dataset(Dataset):
    def __init__(self, data_root, label_csv_path,gene='egfr',mode="train"):
        self.data_root = data_root
        self.label_csv_path = label_csv_path
        self.gene = gene
        self.df = pd.read_csv(self.label_csv_path)
        self.patient_list = []
        self.label_dict = {"Wildtype": 0, "Mutant": 1}
        for patient in os.listdir(self.data_root):
            if patient in self.df["Case ID"].values:
                if gene == "egfr":
                    if mode == 'train':
                        if patient in egfr_train_patient_list:
                            self.patient_list.append(patient)
                    else:
                        if patient in egfr_val_patient_list:
                            self.patient_list.append(patient)
                elif gene == "kras":
                    if mode == 'train':
                        if patient in kras_train_patient_list:
                            self.patient_list.append(patient)
                    else:
                        if patient in kras_val_patient_list:
                            self.patient_list.append(patient)

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        patient = self.patient_list[idx]
        if self.gene == "egfr":
            label = self.df[self.df["Case ID"] == patient]["EGFR mutation status"].values[0]
            label = self.label_dict[label]
        elif self.gene == "kras":
            label = self.df[self.df["Case ID"] == patient]["KRAS mutation status"].values[0]
            label = self.label_dict[label]
        image_path = os.path.join(self.data_root, patient)
        image_array = load_images_to_array(image_path)
        return image_array,label,patient

class NSCLC_PETCT_Dataset(Dataset):
    def __init__(self, data_root, label_csv_path,gene='egfr',mode="train"):
        self.data_root = data_root
        self.label_csv_path = label_csv_path
        self.gene = gene
        self.df = pd.read_csv(self.label_csv_path)
        self.patient_list = []
        self.label_dict = {"Wildtype": 0, "Mutant": 1}
        for patient in os.listdir(os.path.join(self.data_root,'CT')):
            if patient in self.df["Case ID"].values:
                if gene == "egfr":
                    if mode == 'train':
                        if patient in egfr_train_patient_list:
                            self.patient_list.append(patient)
                    else:
                        if patient in egfr_val_patient_list:
                            self.patient_list.append(patient)
                elif gene == "kras":
                    if mode == 'train':
                        if patient in kras_train_patient_list:
                            self.patient_list.append(patient)
                    else:
                        if patient in kras_val_patient_list:
                            self.patient_list.append(patient)
        for patient in self.patient_list:
            if not os.path.exists(os.path.join(self.data_root,'PET',patient)):
                self.patient_list.remove(patient)

    def __len__(self):
        return len(self.patient_list)
    
    def __getitem__(self, idx):
        patient = self.patient_list[idx]
        if self.gene == "egfr":
            label = self.df[self.df["Case ID"] == patient]["EGFR mutation status"].values[0]
            label = self.label_dict[label]
        elif self.gene == "kras":
            label = self.df[self.df["Case ID"] == patient]["KRAS mutation status"].values[0]
            label = self.label_dict[label]
        ct_image_path = os.path.join(self.data_root,'CT',patient)
        pet_image_path = os.path.join(self.data_root,'PET',patient)
        ct_image_array = load_images_to_array(ct_image_path)
        pet_image_array = load_images_to_array_RGB(pet_image_path)
        return ct_image_array,pet_image_array,label,patient

class Whole_NSCLC_CT_Dataset(Dataset):
    def __init__(self, data_root, label_csv_path,gene='egfr'):
        self.data_root = data_root
        self.label_csv_path = label_csv_path
        self.gene = gene
        self.df = pd.read_csv(self.label_csv_path)
        self.patient_list = []
        self.label_dict = {"Wildtype": 0, "Mutant": 1}
        for patient in os.listdir(self.data_root):
            if patient in self.df["Case ID"].values:
                self.patient_list.append(patient)

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        patient = self.patient_list[idx]
        if self.gene == "egfr":
            label = self.df[self.df["Case ID"] == patient]["EGFR mutation status"].values[0]
            label = self.label_dict[label]
        elif self.gene == "kras":
            label = self.df[self.df["Case ID"] == patient]["KRAS mutation status"].values[0]
            label = self.label_dict[label]
        image_path = os.path.join(self.data_root, patient)
        image_array = load_images_to_array(image_path)
        return image_array,label,patient

class Whole_NSCLC_PETCT_Dataset(Dataset):
    def __init__(self, data_root, label_csv_path,gene='egfr'):
        self.data_root = data_root
        self.label_csv_path = label_csv_path
        self.gene = gene
        self.df = pd.read_csv(self.label_csv_path)
        self.label_dict = {"Wildtype": 0, "Mutant": 1}
        self.patient_list = []
        for patient in os.listdir(os.path.join(self.data_root,'CT')):
            if patient in self.df["Case ID"].values:
                self.patient_list.append(patient)
        for patient in self.patient_list:
            if not os.path.exists(os.path.join(self.data_root,'PET',patient)):
                self.patient_list.remove(patient)

    def __len__(self):
        return len(self.patient_list)
    
    def __getitem__(self, idx):
        patient = self.patient_list[idx]
        if self.gene == "egfr":
            label = self.df[self.df["Case ID"] == patient]["EGFR mutation status"].values[0]
            label = self.label_dict[label]
        elif self.gene == "kras":
            label = self.df[self.df["Case ID"] == patient]["KRAS mutation status"].values[0]
            label = self.label_dict[label]
        ct_image_path = os.path.join(self.data_root,'CT',patient)
        pet_image_path = os.path.join(self.data_root,'PET',patient)
        ct_image_array = load_images_to_array(ct_image_path)
        pet_image_array = load_images_to_array_RGB(pet_image_path)
        return ct_image_array,pet_image_array,label,patient
        

                
        
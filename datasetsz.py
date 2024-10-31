import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
def image_normalization(image, contrast):
    """ Normalize the image based on the type of contrast. """
    
    if contrast == 'NCCT':
        # 将小于0的值置为0，将大于100的值置为0。
        image[image < 0] = 0
        image[image > 100] = 0
    elif contrast == 'blood':
        # 二值化：所有标签都变为1
        image[image > 0] = 1  # 假设原来标签为1、2、3，只要值大于0的变为1
        # 如果需要将零值保留为零，可以不做处理
    '''
    else:
        # 对于其他类型的图像，可以在这里添加更多的处理逻辑
        if contrast == 'ADC':
            mask_rmventricle = image < 2000  # 移除小于2000的值
            image *= mask_rmventricle
        value_mean = np.mean(image[np.nonzero(image)])  # 找到非零均值
        image /= value_mean  # 归一化
    '''
    return image


def pad_image(image, target_dim):
    # 创建全零的目标图像
    padding_image = np.zeros(target_dim)
    # 计算填充的边界
    xpadding = (target_dim[0] - image.shape[0]) // 2
    ypadding = (target_dim[1] - image.shape[1]) // 2
    zpadding = (target_dim[2] - image.shape[2]) // 2
    
    # 将原图像放入填充后的图像中
    padding_image[xpadding:xpadding + image.shape[0],
                  ypadding:ypadding + image.shape[1],
                  zpadding:zpadding + image.shape[2]] = image
    return padding_image

class NiftiDataset(Dataset):
    def __init__(self, dataframe, positive_dir, negative_dir,target_dim):
        self.dataframe = dataframe
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.target_dim = target_dim
        self.data = []
        self.labels = []
        self.patient_ids = []
        self._load_data(positive_dir, label=1)
        self._load_data(negative_dir, label=0)

    def _load_data(self, directory, label):
        for patient_folder in os.listdir(directory):
            # Assume patient_folder is the ID and should be a valid folder
            folder_path = os.path.join(directory, patient_folder)
            if os.path.isdir(folder_path):
                patient_id = patient_folder
                if patient_id in self.dataframe['Patient ID'].values:
                    # Load NIfTI files here (adjust according to your data structure)
                    self.data.append(folder_path)  # Or load the actual data
                    self.labels.append(label)
                    self.patient_ids.append(patient_id)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient_path = self.data[idx]
        patient_id = self.patient_ids[idx]
        # Load NIfTI images
        try:
            ncct_img = nib.load(os.path.join(patient_path, 'NCCT.nii.gz')).get_fdata()
            blood_img = nib.load(os.path.join(patient_path, 'blood.nii.gz')).get_fdata()
            # mask_img = nib.load(os.path.join(patient_path, 'NCCTmask.nii.gz')).get_fdata()
        except FileNotFoundError as e:
            print(f"File not found for patient {patient_path}: {e}")
            # 如果文件未找到，返回全零的输入和标签
            input_data = np.zeros((2, *self.target_dim))
            return torch.tensor(input_data, dtype=torch.float32), self.labels[idx], patient_id

        # Normalize and pad images
        ncct_img = image_normalization(ncct_img, 'NCCT')
        blood_img = image_normalization(blood_img, 'blood')
        # mask_img = image_normalization(mask_img, 'mask')

        ncct_img = pad_image(ncct_img, self.target_dim)
        blood_img = pad_image(blood_img, self.target_dim)
        # mask_img = pad_image(mask_img, self.target_dim)

        # Create a combined input
        # input_data = np.stack((ncct_img, blood_img, mask_img), axis=0)
        input_data = np.stack((ncct_img, blood_img), axis=0)
        # Get the label
        label = self.labels[idx]
        
        return torch.tensor(input_data, dtype=torch.float32), label, patient_id


# Directories
positive_dir = '/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/train_nii_close'
negative_dir = '/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/train_nii_nc'

# Create dataset and dataloader
# Create dataset and dataloader
if __name__ == '__main__':
    # Assuming you have a CSV file with Patient ID and Label; for now, using None for the dataframe
    dataframe = None  
    target_dim = (512, 512, 200)
    dataset = NiftiDataset(dataframe, positive_dir, negative_dir, target_dim)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Example to iterate through the data
    for batch_data, batch_labels in data_loader:
        print(batch_data.shape, batch_labels)
# 测试迭代数据加载器
'''
for inputs, labels in data_loader:
    print(inputs.shape, labels.shape)
    break
'''
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def get_patient_data(positive_dir, negative_dir):
    data = []
    
    # Load positive samples
    for patient_folder in os.listdir(positive_dir):
        patient_path = os.path.join(positive_dir, patient_folder)
        if os.path.isdir(patient_path):
            data.append((patient_folder, 1))  # Positive label

    # Load negative samples
    for patient_folder in os.listdir(negative_dir):
        patient_path = os.path.join(negative_dir, patient_folder)
        if os.path.isdir(patient_path):
            data.append((patient_folder, 0))  # Negative label

    return data

def save_data_to_csv(data, output_path):
    df = pd.DataFrame(data, columns=['Patient ID', 'Label'])
    df.to_csv(output_path, index=False)

def main(positive_dir, negative_dir, output_dir):
    # Get all patient data
    patient_data = get_patient_data(positive_dir, negative_dir)

    # Split data into train, validation, and test sets
    train_data, temp_data = train_test_split(patient_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # Split remaining into val and test

    # Save the datasets to CSV files
    save_data_to_csv(train_data, os.path.join(output_dir, 'train_data.csv'))
    save_data_to_csv(val_data, os.path.join(output_dir, 'val_data.csv'))
    save_data_to_csv(test_data, os.path.join(output_dir, 'test_data.csv'))

if __name__ == "__main__":
    positive_dir = '/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/train_nii_close'  # Positive samples directory
    negative_dir = '/home/zsun/NCCT_blood_nii/ncct/Radiology2022/data/train_nii_nc'  # Negative samples directory
    output_dir = '/home/zsun/NCCT_blood_nii/ncct/gpt'  # Output directory for CSV files

    main(positive_dir, negative_dir, output_dir)

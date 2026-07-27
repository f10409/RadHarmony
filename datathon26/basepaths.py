import os

_ROOT = "/mnt/efs/cxr" # "/mnt/NAS4/datasets/external"  # "/mnt/efs/cxr"

MONTGOMERY_DIR = f"{_ROOT}/Montgomery-CXR/MontgomerySet/CXR_png/"

VINDR_ROOT = f"{_ROOT}/VinDr-CXR/1.0.0"
VINDR_DIR = VINDR_ROOT + "/test/"
VINDR_CSV = VINDR_ROOT + "/annotations/image_labels_test.csv"
VINDR_BBOX = VINDR_ROOT + "/annotations/annotations_test.csv"

VQA_RAD_DIR = f"{_ROOT}/VQA-RAD/VQA_RAD Image Folder"

SIIM_DIR = f"{_ROOT}/SIIM_ACR_Pneumothorax/dicom-images-train/"
SIIM_CSV = f"{_ROOT}/SIIM_ACR_Pneumothorax/train-rle.csv"

MIMIC_DIR = f"{_ROOT}/MIMIC-CXR-V2-AWS/files/"

SESSION_DATA = f"{_ROOT}/session_data/"
SESSION_CACHE = f"{_ROOT}/session_cache/"

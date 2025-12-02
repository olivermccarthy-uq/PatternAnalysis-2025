import numpy as np
import glob
import imageio

def load_oasis_data(image_dir, label_dir):
    # Sort image and label files to ensure corresponding order
    image_files = sorted(glob.glob(f"{image_dir}/*.nii.png"))
    label_files = sorted(glob.glob(f"{label_dir}/*.nii.png"))
    
    images = []
    labels = []
    for img_file, lbl_file in zip(image_files, label_files):
        image = imageio.imread(img_file)
        label = imageio.imread(lbl_file)
        images.append(image)
        labels.append(label)
    
    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)
    return images, labels

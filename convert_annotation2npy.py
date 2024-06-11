

import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm

def read_files_from_folder(folder_path, filename):
    file_data = []
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r') as file:
        for line in file:
            values = list(map(float, line.replace("\n", "").split(",")))
            file_data.append(values)
    return np.array(file_data)

def main(args):
    
    
    # Check if output directories exist, create them if they do not
    output_files ="./converted_files/"

    if not os.path.exists(output_files):
      os.makedirs(output_files)

    txt_filename = os.listdir(args.keypoint_path)

    transformed_coord = np.zeros((len(txt_filename), 198))
    coordinate_shape = []

    transformed_images = np.zeros((len(txt_filename), 224, 224, 3))

    for i in tqdm(range(len(txt_filename))):
        image0 = cv2.imread(os.path.join(args.image_path, txt_filename[i][0:-3] + "jpg"))
        
        height, width = image0.shape[:2]
        
        data = read_files_from_folder(args.keypoint_path, txt_filename[i])
        
        data[:, 0] /= width  # Normalize x coordinates
        data[:, 1] /= height  # Normalize y coordinates
        
        flatten_data = data.flatten()
        
        coordinate_shape.append(len(flatten_data))
        transformed_coord[i, 0:len(flatten_data)] = flatten_data
        
        transformed_images[i] = cv2.resize(image0, (224, 224))

    np.save(output_files+"transformed_images.npy", transformed_images)
    np.save(output_files+"transformed_coord.npy", transformed_coord)

    print(f"Processed images saved to: {output_files}")
    print(f"Transformed coordinates saved to: {output_files}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and keypoint annotations.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the images.")
    parser.add_argument('--keypoint_path', type=str, required=True, help="Path to the keypoint annotation files.")
    
    args = parser.parse_args()
    main(args)

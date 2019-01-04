from PIL import Image
from detector import detect_faces
from align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm

source_root = './test' # Modify to your source dir
dest_root = './test_Aligned_112x112' # Modify to your destination dir
reference = get_reference_facial_points(default_square = True)
if not os.path.isdir(dest_root):
    os.mkdir(dest_root)

for subfolder in tqdm(os.listdir(source_root)):
    if not os.path.isdir(os.path.join(dest_root, subfolder)):
        os.mkdir(os.path.join(dest_root, subfolder))
    for image_name in os.listdir(os.path.join(source_root, subfolder)):
            print("Processing\t{}".format(os.path.join(source_root, subfolder, image_name)))
            img = Image.open(os.path.join(source_root, subfolder, image_name))
            try: # Handle exception
                _, landmarks = detect_faces(img)
            except Exception:
                print("{} is discarded due to exception!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            if len(landmarks) == 0: # If the landmarks cannot be detected, the img will be discarded
                print("{} is discarded due to non-detected landmarks!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(112, 112)) # Modidy the crop_size according to your case
            img_warped = Image.fromarray(warped_face)
            img_warped.save(os.path.join(dest_root, subfolder, image_name))
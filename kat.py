from dino_vit_features import keypoint_utils
#import mediapy as media
import numpy as np
import cv2
import os
def process_frames(input_folder, c = 1, save_visual = True):
    """Processes every c-th frame in the input folder and saves JSON output in the output folder."""
    
    points = [] #[[x1s, y1s], [x2s, y2s]]
    
    # Get all image files in the folder (sorted to maintain order)
    image_files = sorted([
        f for f in os.listdir(input_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    sample0_path = os.path.join(input_folder, image_files[0])
    sample1_path = os.path.join(input_folder, image_files[1])
    patches_xy, desc1, descriptor_vectors, num_patches = keypoint_utils.extract_descriptors(sample0_path, sample1_path, num_pairs = 5, load_size = 480)
    # Process every c-th frame
    for i in range(1, len(image_files), c):
        image_filename = image_files[i]
        image_path = os.path.join(input_folder, image_filename)

        # Call function to process frame
        ys, xs = extract_keypoints(image_path, descriptor_vectors, num_patches)
        points.append([ys, xs])
        img = cv2.imread(image_path)
        img = cv2.resize(img, [480,480])
        output_path = os.path.join(input_folder, "A" + image_filename)
        for x1, y1 in zip(xs, ys):
            cv2.circle(img, (int(x1), int(y1)), radius=4, color=(0, 255, 0), thickness=-1)
        cv2.imwrite(output_path, img)
    return points

def extract_keypoints(img_path, descriptor_vectors, num_patches):
    map, _ = keypoint_utils.extract_desc_maps(img_path)
    ys, xs = keypoint_utils.extract_descriptor_nn(descriptor_vectors, map[0], num_patches, False)
    return ys, xs

process_frames("move_cup")
# img_path1 = "move_cup/frame_0000.jpg"
# img_path2 = "move_cup/frame_0001.jpg"
# img_folder = "move_cup"



# img1 = cv2.resize(cv2.imread(img_path1), [480,480])
# img2 = cv2.resize(cv2.imread(img_path2), [480,480])
# patches_xy, desc1, descriptor_vectors, num_patches = keypoint_utils.extract_descriptors(img_path1, img_path2, num_pairs = 5, load_size = 480)
# map1, _ = keypoint_utils.extract_desc_maps(img_path1)
# map2, _ = keypoint_utils.extract_desc_maps(img_path2)
# y1s, x1s = keypoint_utils.extract_descriptor_nn(descriptor_vectors, map1[0], num_patches, False)
# y2s, x2s = keypoint_utils.extract_descriptor_nn(descriptor_vectors, map2[0], num_patches, False)


# for x1, y1 in zip(x1s, y1s):
#     cv2.circle(img1, (int(x1), int(y1)), radius=4, color=(0, 255, 0), thickness=-1)

# for x2, y2 in zip(x2s, y2s):
#     cv2.circle(img2, (int(x2), int(y2)), radius=4, color=(0, 0, 255), thickness=-1)

# if img1.shape[0] != img2.shape[0]:
#     # Resize img2 to match height of img1
#     img2 = cv2.resize(img2, (int(img2.shape[1] * img1.shape[0] / img2.shape[0]), img1.shape[0]))

# for i in range(1,10):
#     map_i, _ = keypoint_utils.extract_desc_maps(img_path1)

# combined = np.hstack((img1, img2))
# cv2.imshow("Keypoints on Image1 (green) and Image2 (red)", combined)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
pass
#[array([31, 22, 15, 33, 29], dtype=int64), array([75, 75, 37, 78, 81], dtype=int64), array([30, 20, 15, 32, 28], dtype=int64), array([59, 59, 89, 62, 65], dtype=int64)]
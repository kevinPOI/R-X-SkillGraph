# Fine-grained Breakfast dataset

This repository provides resources for evaluating action label predictions on videos from the Fine-grained Breakfast dataset. It includes ground-truth annotations and an evaluation script.

This dataset is provided as supplementary material for the paper:

> **Open-vocabulary action localization with iterative visual prompting**  
> *Naoki Wake, Atsushi Kanehira, Kazuhiro Sasabuchi, Jun Takamatsu, Katsushi Ikeuchi (2024)*  
> [arXiv:2408.17422](https://arxiv.org/abs/2408.17422)
>
> ```bibtex
> @article{wake2024open,
>   title={Open-vocabulary action localization with iterative visual prompting},
>   author={Wake, Naoki and Kanehira, Atsushi and Sasabuchi, Kazuhiro and Takamatsu, Jun and Ikeuchi, Katsushi},
>   journal={arXiv preprint arXiv:2408.17422},
>   year={2024}
> }
> ```

The original data is derived from the dataset described below. We have manually annotated a subset of these videos:

> **Human grasping database for activities of daily living with depth, color and kinematic data streams**  
> *Artur Saudabayev, Zhanibek Rysbek, Raykhan Khassenova, Huseyin Atakan Varol (2018)*  
> *Scientific Data, 5(1), 1–13*
>
> ```bibtex
> @article{saudabayev2018human,
>   title={Human grasping database for activities of daily living with depth, color and kinematic data streams},
>   author={Saudabayev, Artur and Rysbek, Zhanibek and Khassenova, Raykhan and Varol, Huseyin Atakan},
>   journal={Scientific data},
>   volume={5},
>   number={1},
>   pages={1--13},
>   year={2018},
>   publisher={Nature Publishing Group}
> }
> ```

## Directory and File Structure

- **original_videos**  
  Download the original videos from  `Human grasping database for activities of daily living with depth, color and kinematic data streams` and place them in this folder.
  
- **label_data_gt_right.json**  
  This JSON file holds the ground-truth annotations for the videos. Each entry in the JSON contains:
  - **action**: A sequence of action labels that occur in the video.  
    *Example*: `["Grasp with the right hand", "Picking with the right hand", ...]`
  - **gt_time**: The frame index annotations corresponding to each action label.  
    *Example*: `[[0, 23], [24, 48], ...]`
  - **video_path**: The relative path to the corresponding video file.  
    *Example*: `"original_videos\\subject_9_gopro_seg_1_2324-2575.mp4"`  
    **Note**: This file name is constructed from the original video name with the appended frame range. Since this repository does not provide the original videos, you need to download the original dataset, extract the clips corresponding to the specified frame numbers, and place them in the `original_videos` folder. We provide the script `clip_original_videos.py` to extract these clips. The list of original video files is provided in `original_videos\\original_videos.txt`.

- **compute_mof_iou_f1.py**  
  This evaluation script computes performance metrics (e.g., MOF, IoU, and F1 score) by comparing predicted action labels with the ground truth.

- **label_data_estimate_baseline.json**  
  This is an example file that contains estimated action labels. It is used as an input to the evaluation script.

- **clip_original_videos.py**  
  This script extracts video clips from the original videos based on the frame indices specified in `label_data_gt_right.json`. Running this script will generate the video dataset with filenames as indicated in the JSON annotations.

## Usage Instructions

1. **Place the Video Files**  
   - Download the original videos from the Fine-grained Breakfast dataset.  
   - Place the downloaded video files in the `original_videos` folder. Refer to `original_videos\\original_videos.txt` for the list of required files.  

2. **Generate the Video Dataset**  
   After placing the original videos in the `original_videos` folder, run the `clip_original_videos.py` script to extract the annotated clips. This script uses the frame index annotations provided in `label_data_gt_right.json` to cut the clips from the original videos and save them using the specified naming convention. Run the script with the following command. Note that this script leverages ffmpeg.
   ```bash
   python clip_original_videos.py
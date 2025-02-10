import os
import json
import re
import subprocess

# Configuration
json_file = "label_data_gt_right.json"  # Path to the JSON file
frame_rate = 30  # Frame rate of the videos

# Read the JSON file
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Process each entry in the JSON file
for entry in data:
    # Get the video_path field (expected format, e.g., "original_videos/subject_1_gopro_seg_1_2162-2284.mp4")
    video_path_field = entry.get("video_path")
    if not video_path_field:
        print("Missing video_path field. Skipping entry.")
        continue

    # Normalize path separators (in case backslashes are used)
    video_path_field = video_path_field.replace("\\", "/")
    
    # Split the video_path into directory and filename
    directory, filename = os.path.split(video_path_field)
    basename, ext = os.path.splitext(filename)
    
    # Use regex to extract segment info from the end of the basename.
    # Expected pattern: an underscore followed by two numbers separated by a dash, e.g., "_2162-2284"
    match = re.search(r'_(\d+)-(\d+)$', basename)
    if not match:
        print(f"Segment info not found in filename: {filename}. Skipping entry.")
        continue

    segment_start_str, segment_end_str = match.groups()
    try:
        segment_start_frame = int(segment_start_str)
        segment_end_frame = int(segment_end_str)
    except ValueError:
        print(f"Invalid segment frame numbers in filename: {filename}. Skipping entry.")
        continue

    # Calculate the duration in frames and verify the range is valid
    duration_frames = segment_end_frame - segment_start_frame
    if duration_frames <= 0:
        print(f"Invalid frame range in filename: {filename}. Skipping entry.")
        continue

    # Convert frame numbers to seconds for FFmpeg
    start_time_sec = segment_start_frame / frame_rate
    duration_sec = duration_frames / frame_rate

    # Determine the original video filename by removing the segment info from the basename.
    # For example, if basename is "subject_1_gopro_seg_1_2162-2284", the original basename will be "subject_1_gopro_seg_1".
    original_basename = basename[:match.start()]  # Everything before the segment info
    original_filename = original_basename + ext
    original_video_path = os.path.join(directory, original_filename)
    
    if not os.path.exists(original_video_path):
        print(f"Original video file not found: {original_video_path}. Skipping entry.")
        continue

    # Use FFmpeg to extract the clip from the original video.
    # The clip starts at 'start_time_sec' (in seconds) and lasts for 'duration_sec' seconds.
    # The output file will be saved with the same name as specified in the video_path field.
    output_video_path = video_path_field
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", original_video_path,
        "-ss", str(start_time_sec),
        "-t", str(duration_sec),
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "128k",
        "-y", output_video_path
    ]

    print(f"Extracting clip: {output_video_path}")
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("All extractions completed!")

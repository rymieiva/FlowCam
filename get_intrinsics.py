import gzip
import json
import os

# Path to the annotation file
annotation_file = "C:\\Users\\rymi\\work\\FlowCam\\hydrant_flowcam\\hydrant\\frame_annotations.jgz"  # or sequence_annotations.jgz

# Load the gzipped JSON file
with gzip.open(annotation_file, "rt", encoding="utf-8") as f:
    data = json.load(f)  # This loads the data as a list of dictionaries

# Since data is a list, let's print the first element to see its structure
print(data[0])  # To inspect the structure of a single entry

# Example: Extract intrinsics for a specific frame
frame_id = "106_12648_23157"
frame_data = None

# Iterate through the list to find the frame by its sequence_name
for entry in data:
    if entry.get("sequence_name") == frame_id:
        frame_data = entry
        break

if frame_data is None:
    raise ValueError(f"Frame {frame_id} not found in the annotations.")

# Extract intrinsic parameters if the frame data is found
viewpoint_data = frame_data.get("viewpoint", {})
fx, fy = viewpoint_data.get("focal_length", [None, None])
cx, cy = viewpoint_data.get("principal_point", [None, None])

# Check if all values were successfully extracted
if None in [fx, fy, cx, cy]:
    raise ValueError(f"Could not extract all intrinsic parameters for frame {frame_id}.")

# Print the extracted intrinsic parameters
intrinsics = f"{fx},{fy},{cx},{cy}"
print(f"Intrinsics for frame {frame_id}: {intrinsics}")
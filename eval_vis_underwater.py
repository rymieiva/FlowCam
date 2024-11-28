import os
from PIL import Image, ImageDraw, ImageFont

# Paths
output_dir = ".\\visualizations\\10cat"
image_dir = ".\\underwater-10cat-320x320\\output"
metrics_file = ".\\underwater-10cat-320x320\\output\\psnr&lpips.txt"

# Configurations
output_res = (1920, 1080)  # Resolution of the output visualization image
font_path = ".\\arial.ttf"  # Path to a TrueType font file (e.g., Arial.ttf)

# Number of rows and columns in the visualization
rows = 3
columns = 6  # Depth, Est, GT for each sequence

# Adjusting sequence image size
image_width = (output_res[0] // columns) + 240  # Increased width for further stretching
image_height = output_res[1] // rows
adjusted_res = (image_width, image_height)

def load_metrics(metrics_file):
    """Load PSNR and LPIPS values from the text file."""
    metrics = []
    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("psnr"):
                continue
            try:
                parts = line.strip().split(",")
                psnr = float(parts[0].split(" ")[1])
                lpips = float(parts[1].split(" ")[2])
                eval_idx = int(parts[2].split(" ")[-1])
                metrics.append((eval_idx, psnr, lpips))
            except (IndexError, ValueError) as e:
                print(f"Skipping malformed line: {line.strip()} - {e}")
    return metrics

def create_visualization(image_dir, metrics, output_res, rows, columns, output_dir):
    """Create visualizations for depth, estimated, and GT images."""
    os.makedirs(output_dir, exist_ok=True)
    sequence_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    # Group images by sequence
    sequences = {}
    for file in sequence_files:
        if "_depth" in file or "_est" in file or "_gt" in file:
            seq_id = int(file.split("_")[0])  # Extract sequence ID
            if seq_id not in sequences:
                sequences[seq_id] = {}
            if "_depth" in file:
                sequences[seq_id]["depth"] = file
            elif "_est" in file:
                sequences[seq_id]["est"] = file
            elif "_gt" in file:
                sequences[seq_id]["gt"] = file

    # Sort sequences by sequence ID
    sorted_sequences = sorted(sequences.items())

    # Visualization
    sequences_per_image = rows  # Rows correspond to sequences
    for start_idx in range(0, len(sorted_sequences), sequences_per_image):
        end_idx = start_idx + sequences_per_image
        batch = sorted_sequences[start_idx:end_idx]

        # Create a blank canvas
        canvas = Image.new("RGB", output_res, color="white")
        draw = ImageDraw.Draw(canvas)

        # Font setup
        try:
            font = ImageFont.truetype(font_path, 20)
        except IOError:
            print("Font not found. Using default font.")
            font = ImageFont.load_default()

        # Populate canvas with images and metrics
        for seq_idx, (seq_id, files) in enumerate(batch):
            y_offset = seq_idx * image_height

            # Load depth, est, and GT images
            try:
                depth_img = Image.open(os.path.join(image_dir, files["depth"])).resize(adjusted_res)
                est_img = Image.open(os.path.join(image_dir, files["est"])).resize(adjusted_res)
                gt_img = Image.open(os.path.join(image_dir, files["gt"])).resize(adjusted_res)
            except KeyError as e:
                print(f"Missing file for sequence {seq_id}: {e}")
                continue

            # Paste images into the canvas
            canvas.paste(depth_img, (0, y_offset))
            canvas.paste(est_img, (adjusted_res[0], y_offset))
            canvas.paste(gt_img, (adjusted_res[0] * 2, y_offset))

            # Add PSNR, LPIPS, and eval_idx text
            psnr, lpips = next(((psnr, lpips) for idx, psnr, lpips in metrics if idx == seq_id), (None, None))
            text = (
                f"id={seq_id},\n PSNR={psnr:.2f},\n LPIPS={lpips:.4f}"
                if psnr and lpips
                else f"Eval_idx={seq_id}: No Metrics"
            )
            draw.text(
                (adjusted_res[0] * 3 + 10, y_offset + adjusted_res[1] // 3),
                text,
                fill="black",
                font=font,
                stroke_width=1,
                stroke_fill="white"
            )

        # Save the visualization
        output_path = os.path.join(output_dir, f"vis_{start_idx // sequences_per_image}.png")
        canvas.save(output_path)
        print(f"Saved: {output_path}")

# Load metrics
metrics = load_metrics(metrics_file)

# Create visualizations
create_visualization(image_dir, metrics, output_res, rows, columns, output_dir)

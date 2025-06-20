import os 
import json 
from PIL import Image, ImageDraw

INPUT_JSON_DIR = "g_predictions"
INPUT_IMAGE_DIR = "g_dataset"
OUTPUT_PIC_DIR = "g_predictions_png"

os.makedirs(OUTPUT_PIC_DIR, exist_ok=True)

def draw_trex_boxes(image_path, trex_json, save_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    objects = trex_json.get("objects", [])
    if not objects:
        print(f"⚠ No detections for {os.path.basename(image_path)}")
    
    for obj in objects:
        bbox = [int(x) for x in obj["bbox"]]
        score = obj["score"]
        draw.rectangle(bbox, outline="red", width=3)
        draw.text((bbox[0], bbox[1] - 10), f"{score:.2f}", fill="red")
        
    image.save(save_path)
    print(f"✅ Visual saved: {save_path}")

# Loop through JSONs
for json_file in os.listdir(INPUT_JSON_DIR):
    if json_file.endswith(".json"):
        json_path = os.path.join(INPUT_JSON_DIR, json_file)
        basename = os.path.splitext(json_file)[0]
        image_path = os.path.join(INPUT_IMAGE_DIR, f"{basename}.png")
        output_path = os.path.join(OUTPUT_PIC_DIR, f"{basename}.png")

        with open(json_path) as f:
            trex_json = json.load(f)

        draw_trex_boxes(image_path, trex_json, output_path)
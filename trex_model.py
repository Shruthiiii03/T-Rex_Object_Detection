import os
import json
import xml.etree.ElementTree as ET 
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import V2Task, create_task_with_local_image_auto_resize
from PIL import Image, ImageDraw


# Config 
# TOKEN = "YOUR TOKEN" 
IMAGE_LIST = [
    os.path.join("g_dataset", f)
    for f in os.listdir("g_dataset")
    if f.endswith(".png")
]

API_PATH = "/v2/task/trex/detection"
OUTPUT_JSON_DIR = "g_predictions"

# Setup
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

# Helper: Extract first bollard rect from XML
def get_first_bollard_rect(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find('object')
        if obj is not None:
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            return [xmin, ymin, xmax, ymax]
    except Exception as e:
        print(f"Error reading {xml_path}: {e}")
    return None


# init client 
config = Config(TOKEN)
client = Client(config)

# Loop
for image_path in IMAGE_LIST:
    print(f"Processing {image_path}...")
    basename = os.path.splitext(os.path.basename(image_path))[0]
    xml_path = os.path.join("g_groundtruth", f"{basename}.xml")
    rect = get_first_bollard_rect(xml_path)

    if rect is None:
        print(f"⚠ No valid bounding box found for {basename}, skipping...")
        continue

    api_body = {
        "model": "T-Rex-2.0",
        "targets": ["bbox"],
        "bbox_threshold": 0.25,
        "iou_threshold": 0.8,
        "prompt": {
            "type": "visual_images",
            "visual_images": [
                {
                    "interactions": [
                        {
                            "type": "rect",
                            "category_id": 1,
                            "rect": rect
                        }
                    ]
                }
            ]
        }
    }

    # Create task
    task = create_task_with_local_image_auto_resize(
        api_path=API_PATH,
        api_body_without_image=api_body,
        image_path=image_path,
    )

    # Run task
    client.run_task(task)

    # Save result
    json_path = os.path.join(OUTPUT_JSON_DIR, f"{basename}.json")
    with open(json_path, "w") as f:
        json.dump(task.result, f)

    print(f"Saved: {json_path}")

    # Save PNG
    # vis_path = os.path.join(OUTPUT_JSON_DIR, f"{basename}.png")
    # draw_trex_boxes(image_path, task.result, vis_path)
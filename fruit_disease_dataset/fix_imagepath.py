import os
import json

ANN_DIR = "Anthracnose_augmented/annotations"

for file in os.listdir(ANN_DIR):
    if not file.endswith(".json"):
        continue

    path = os.path.join(ANN_DIR, file)

    with open(path, "r") as f:
        data = json.load(f)

    image_name = file.replace(".json", ".jpg")
    data["imagePath"] = image_name

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

print("imagePath fixed for all JSON files ✅")
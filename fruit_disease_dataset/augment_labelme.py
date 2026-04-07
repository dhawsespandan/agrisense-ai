import os
import json
import cv2
import numpy as np

IMAGE_DIR = "Anthracnose/images"
ANNOTATION_DIR = "Anthracnose/annotations"

OUTPUT_IMAGE_DIR = "Anthracnose_augmented/images"
OUTPUT_ANN_DIR = "Anthracnose_augmented/annotations"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_ANN_DIR, exist_ok=True)


def rotate_image_and_points(image, points, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, M, (w, h))

    rotated_points = []
    for x, y in points:
        px = M[0, 0]*x + M[0, 1]*y + M[0, 2]
        py = M[1, 0]*x + M[1, 1]*y + M[1, 2]
        rotated_points.append([float(px), float(py)])

    return rotated_img, rotated_points


def flip_image_and_points(image, points):
    flipped_img = cv2.flip(image, 1)
    h, w = image.shape[:2]

    flipped_points = []
    for x, y in points:
        flipped_points.append([float(w - x), float(y)])

    return flipped_img, flipped_points


def adjust_brightness(image, factor=1.2):
    bright = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return bright


for filename in os.listdir(IMAGE_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    json_path = os.path.join(ANNOTATION_DIR, filename.replace(".JPG", ".json").replace(".jpg", ".json"))

    if not os.path.exists(json_path):
        continue

    image = cv2.imread(image_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    shapes = data["shapes"]

    # Save original
    cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, filename), image)
    with open(os.path.join(OUTPUT_ANN_DIR, filename.replace(".JPG", ".json")), 'w') as f:
        json.dump(data, f, indent=2)

    # AUGMENT 1: Rotate
    for shape in shapes:
        rotated_img, rotated_points = rotate_image_and_points(image, shape["points"], 15)

        new_data = data.copy()
        new_data["imagePath"] = filename.replace(".JPG", "_rot.jpg")
        new_data["shapes"] = [{
            "label": shape["label"],
            "points": rotated_points,
            "shape_type": "polygon",
            "flags": {},
            "group_id": None,
            "description": "",
            "difficult": False,
            "attributes": {},
            "kie_linking": []
        }]

        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, filename.replace(".JPG", "_rot.jpg")), rotated_img)
        with open(os.path.join(OUTPUT_ANN_DIR, filename.replace(".JPG", "_rot.json")), 'w') as f:
            json.dump(new_data, f, indent=2)

    # AUGMENT 2: Flip
    for shape in shapes:
        flipped_img, flipped_points = flip_image_and_points(image, shape["points"])

        new_data = data.copy()
        new_data["imagePath"] = filename.replace(".JPG", "_flip.jpg")
        new_data["shapes"] = [{
            "label": shape["label"],
            "points": flipped_points,
            "shape_type": "polygon",
            "flags": {},
            "group_id": None,
            "description": "",
            "difficult": False,
            "attributes": {},
            "kie_linking": []
        }]

        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, filename.replace(".JPG", "_flip.jpg")), flipped_img)
        with open(os.path.join(OUTPUT_ANN_DIR, filename.replace(".JPG", "_flip.json")), 'w') as f:
            json.dump(new_data, f, indent=2)

    # AUGMENT 3: Brightness
    bright_img = adjust_brightness(image, 1.3)
    cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, filename.replace(".JPG", "_bright.jpg")), bright_img)
    with open(os.path.join(OUTPUT_ANN_DIR, filename.replace(".JPG", "_bright.json")), 'w') as f:
        json.dump(data, f, indent=2)

print("Augmentation Complete ✅")
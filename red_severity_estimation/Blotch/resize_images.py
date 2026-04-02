from PIL import Image
import os

input_folder = r"C:\Users\Anindita\OneDrive\Desktop\Apple_Disease_Project\Blotch\images"
output_folder = r"C:\Users\Anindita\OneDrive\Desktop\Apple_Disease_Project\Blotch\images_resized"

os.makedirs(output_folder, exist_ok=True)

target_width = 1200
count = 0

for file in os.listdir(input_folder):
    img_path = os.path.join(input_folder, file)
    try:
        img = Image.open(img_path).convert("RGB")
        new_height = int(target_width * img.height / img.width)
        img = img.resize((target_width, new_height), Image.LANCZOS)
        img.save(os.path.join(output_folder, file))
        print("Resized:", file)
        count += 1
    except Exception as e:
        print("Skipped:", file, "|", e)

print("TOTAL RESIZED:", count)
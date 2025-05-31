import os
from PIL import Image
from tqdm import tqdm
import json

root_dir = "./data"
source_images_dir = "./data_raw/FinalTargetData512"
image_dir_in_data = "images"  # ./data/images 에 저장될 것임
captions_dir = "captions"

images_output_dir_absolute = os.path.join(root_dir, image_dir_in_data)
captions_dir_absolute = os.path.join(root_dir, captions_dir)

# 출력 디렉토리 생성
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
if not os.path.exists(images_output_dir_absolute):
    os.makedirs(images_output_dir_absolute)
if not os.path.exists(captions_dir_absolute):
    os.makedirs(captions_dir_absolute)

image_format = "png"
json_name = "partition/data_info.json"
partition_dir = os.path.join(root_dir, "partition")
if not os.path.exists(partition_dir):
    os.makedirs(partition_dir)

absolute_json_name = os.path.join(root_dir, json_name)
data_info = []

# 원본 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(source_images_dir) if f.endswith(f'.{image_format}')]

order = 0
for image_filename in tqdm(image_files):
    source_image_path = os.path.join(source_images_dir, image_filename)
    
    try:
        image = Image.open(source_image_path)
        # 이미지를 ./data/images/ 경로에 원본 파일명 그대로 저장 (또는 order 기반으로 저장도 가능)
        # 여기서는 order 기반으로 저장하고, data_info.json에는 새로운 파일명을 기록합니다.
        output_image_filename = f"{order}.{image_format}"
        output_image_path_absolute = os.path.join(images_output_dir_absolute, output_image_filename)
        image.save(output_image_path_absolute)
        
        # 캡션 파일은 생성하지 않음 (주석 처리된 부분 유지)
        # with open(f"{captions_dir_absolute}/{order}.txt", "w") as text_file:
        #     text_file.write("...")
        
        width, height = image.size # 원본 이미지 크기 사용
        # 만약 모든 이미지를 512x512로 강제하고 싶다면 아래 주석 해제
        # width, height = 512, 512 
        # if image.size != (512,512):
        #     image = image.resize((512,512))
        #     image.save(output_image_path_absolute) # 리사이즈 후 다시 저장
            
        ratio = width / height # 비율 계산
        
        data_info.append({
            "height": height,
            "width": width,
            "ratio": ratio,
            "path": f"{image_dir_in_data}/{output_image_filename}", # ./data/images/0.png 형태
            "prompt": "A high-quality, photorealistic image of a healthy, anatomically normal human baby, reconstructed solely from the fetal shape and pose of the input 3D ultrasound image of an fetal. ",
        })
            
        order += 1
    except Exception as e:
        print(f"Error processing {source_image_path}: {e}")

with open(absolute_json_name, "w") as json_file:
    json.dump(data_info, json_file)

print(f"Processed {order} images and created {absolute_json_name}")
import onnxruntime as ort
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os

# Load ONNX models
sess_smile = ort.InferenceSession("/Users/deniskrylov/Developer/University/compose-glide/models/binary_classifiers/resnet18_smiling_model.onnx")
sess_glasses = ort.InferenceSession("/Users/deniskrylov/Developer/University/compose-glide/models/binary_classifiers/resnet18_eyeglasses_model.onnx")
sess_gender = ort.InferenceSession("/Users/deniskrylov/Developer/University/compose-glide/models/binary_classifiers/resnet18_male_model.onnx")

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def predict(img, session):
    inputs = {session.get_inputs()[0].name: img}
    out = session.run(None, inputs)[0]
    return np.argmax(out)

# Target labels
prompt_targets = {
    0: {"smile": 0, "glasses": 0, "gender": 1},
    1: {"smile": 1, "glasses": 1, "gender": 1},
    2: {"smile": 1, "glasses": 0, "gender": 0},
    3: {"smile": 1, "glasses": 1, "gender": 0},
    4: {"smile": 0, "glasses": 0, "gender": 0},
}

# Accuracy trackers
smile_correct = [0] * 5
glasses_correct = [0] * 5
gender_correct = [0] * 5
total = [0] * 5

# Output directory
output_dir = "/Users/deniskrylov/Developer/University/compose-glide/outputs"
print(f"Looking for images in: {output_dir}")

for i in range(0, 5):  # Range from 1 to 5 to match prompt_targets keys
    # Use glob to get all matching files
    pattern = os.path.join(output_dir, f"prompt_{i}_variant_*.png")
    files = glob(pattern)
    
    print(f"Found {len(files)} files for prompt {i}: {pattern}")
    
    target = prompt_targets[i]
    idx = i-1  # Index for our tracking arrays (0-4)

    for file in tqdm(files, desc=f"Prompt {i}"):
        try:
            img = preprocess(file)
            pred_smile = predict(img, sess_smile)
            pred_glasses = predict(img, sess_glasses)
            pred_gender = predict(img, sess_gender)

            if pred_smile == target["smile"]:
                smile_correct[idx] += 1
            if pred_glasses == target["glasses"]:
                glasses_correct[idx] += 1
            if pred_gender == target["gender"]:
                gender_correct[idx] += 1

            total[idx] += 1
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Display per-component accuracy
print("\n📊 Component-wise Accuracy per Prompt:")
for i in range(5):
    if total[i] > 0:
        smile_acc = 100 * smile_correct[i] / total[i]
        glasses_acc = 100 * glasses_correct[i] / total[i]
        gender_acc = 100 * gender_correct[i] / total[i]
        print(f"Prompt {i+1}:")
        print(f"  - Smile Accuracy   : {smile_acc:.2f}%")
        print(f"  - Glasses Accuracy : {glasses_acc:.2f}%")
        print(f"  - Gender Accuracy  : {gender_acc:.2f}%")
    else:
        print(f"Prompt {i+1}: No valid images processed")

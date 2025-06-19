import onnxruntime as ort
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm


# Load ONNX models
sess_smile = ort.InferenceSession("/content/resnet18_smiling_model.onnx")
sess_glasses = ort.InferenceSession("/content/resnet18_eyeglasses_model.onnx")
sess_gender = ort.InferenceSession("/content/resnet18_male_model.onnx")

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
    1: {"smile": 0, "glasses": 0, "gender": 1},
    2: {"smile": 1, "glasses": 1, "gender": 1},
    3: {"smile": 1, "glasses": 0, "gender": 0},
    4: {"smile": 1, "glasses": 1, "gender": 0},
    5: {"smile": 1, "glasses": 1, "gender": 0},
}

# Accuracy trackers
smile_correct = [0] * 5
glasses_correct = [0] * 5
gender_correct = [0] * 5
total = [0] * 5

for i in range(1, 6):
    files = glob(f"/Users/deniskrylov/Developer/University/compose-glide/outputs/prompt_{i}variant*.png")
    target = prompt_targets[i]

    for file in tqdm(files, desc=f"Prompt {i}"):
        img = preprocess(file)
        pred_smile = predict(img, sess_smile)
        pred_glasses = predict(img, sess_glasses)
        pred_gender = predict(img, sess_gender)

        if pred_smile == target["smile"]:
            smile_correct[i-1] += 1
        if pred_glasses == target["glasses"]:
            glasses_correct[i-1] += 1
        if pred_gender == target["gender"]:
            gender_correct[i-1] += 1

        total[i-1] += 1

# Display per-component accuracy
print("\n📊 Component-wise Accuracy per Prompt:")
for i in range(5):
    smile_acc = 100 * smile_correct[i] / total[i]
    glasses_acc = 100 * glasses_correct[i] / total[i]
    gender_acc = 100 * gender_correct[i] / total[i]
    print(f"Prompt {i+1}:")
    print(f"  - Smile Accuracy   : {smile_acc:.2f}%")
    print(f"  - Glasses Accuracy : {glasses_acc:.2f}%")
    print(f"  - Gender Accuracy  : {gender_acc:.2f}%")

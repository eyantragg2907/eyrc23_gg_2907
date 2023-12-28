from sys import platform
import numpy as np
import subprocess
import shutil
import ast
import sys
import os

import torch
import torchvision

classmap = [
    "combat",
    "destroyed_buildings",
    "fire",
    "human_aid_rehabilitation",
    "military_vehicles",
]

MODEL_PATH = "pjr_model_overfit_75_20231228_124401.pt"
# IMAGE_PATH = "pjrtest/0/0.png"
IMAGE_DIR = "pjrtest/4"

model = torch.jit.load(MODEL_PATH, map_location="cpu")  # type: ignore
model.eval()
with torch.inference_mode():

    for path in os.listdir(IMAGE_DIR):
        if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg"):
            imagepath = os.path.join(IMAGE_DIR, path) 
            print(imagepath)
            img = torchvision.io.read_image(imagepath).float()
            img = torchvision.transforms.Compose(
                [torchvision.transforms.Resize((75, 75), antialias=True)]  # type: ignore
            )(img) 
            x = torch.unsqueeze(img, 0)
            y_pred = model(x)

            class_val = int(y_pred.argmax(dim=1)[0])
            class_pred = classmap[class_val]

            print(f"Predicted class: {class_pred}")

# def model_load():
#     modelpath = "pjr_model_overfit_75_20231228_124401.pt"
#     model = torch.jit.load(modelpath, map_location="cpu")  # type: ignore
#     model.eval()
#     return model

# def predict(model, imagepath):
#     model = model_load()
#     img = torchvision.io.read_image(imagepath).float()
#     img = torchvision.transforms.Compose([torchvision.transforms.Resize((75, 75), antialias=True)])(img) 
#     x = torch.unsqueeze(img, 0)
#     with torch.inference_mode():
#         y_pred = model(x)

#     class_val = int(y_pred.argmax(dim=1)[0])
#     class_pred = classmap[class_val]

#     return class_pred
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

img_path = "./img_cat.jpg"
img = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(img).unsqueeze(0)

model = models.resnet50(pretrained=True)
model.eval()

feature_dict = {}

def get_hook(name):
    def hook_fn(module, input, output):
        feature_dict[name] = output

    return hook_fn

model.conv1.register_forward_hook(get_hook("l0"))
model.layer1.register_forward_hook(get_hook("l1"))
model.layer2.register_forward_hook(get_hook("l2"))
model.layer3.register_forward_hook(get_hook("l3"))
model.layer4.register_forward_hook(get_hook("l4"))

with torch.no_grad():
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

params = list(model.parameters())
weight_softmax = params[-2]
class_weights = weight_softmax[pred_class].squeeze().detach().numpy()

for layer_name, feature in feature_dict.items():
    feat = feature.squeeze(0).detach().numpy()
    cam = np.zeros(feat.shape[1:], dtype=np.float32)
    for i, w in enumerate(class_weights[:feat.shape[0]]):
        cam += w * feat[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (img.width, img.height))
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(img), 0.5, heatmap, 0.5, 0)

    out_path = os.path.join('./cnn', f"{layer_name}.jpg")
    plt.imsave(out_path, superimposed_img[:, :, ::-1])
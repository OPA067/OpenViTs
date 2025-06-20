import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
# layer_bottle_conv: l4_b2_c3
final_conv_layer = model.layer4[2].conv3
feature_maps = []

def hook_fn(module, input, output):
    feature_maps.append(output)

hook = final_conv_layer.register_forward_hook(hook_fn)

output = model(input_tensor)
pred_class = output.argmax(dim=1)

params = list(model.parameters())
weight_softmax = params[-2]
class_weights = weight_softmax[pred_class].squeeze().detach().numpy()

features = feature_maps[0].squeeze().detach().numpy()   # [C, H, W] = [2048, 7, 7]

cam = np.zeros(features.shape[1:], dtype=np.float32)    # [H, W] = [7, 7]
for i, w in enumerate(class_weights):                   # 按照通道进行加权求和
    cam += w * features[i]

cam = np.maximum(cam, 0)
cam = cv2.resize(cam, (img.width, img.height))
cam = (cam - cam.min()) / (cam.max() - cam.min())

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(np.array(img), 0.5, heatmap, 0.5, 0)
plt.imshow(superimposed_img[:, :, ::-1])
plt.axis('off')
plt.savefig("./cnn/cat_l4_b2_c3.jpg", dpi=300)

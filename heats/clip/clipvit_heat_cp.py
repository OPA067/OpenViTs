import torch
import clip
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

def overlay_heatmap_on_image(image, heatmap):
    image_resized = image.resize((224, 224))
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)
    heatmap_uint8 = np.uint8(255 * heatmap_norm)
    heatmap_resized = cv2.resize(heatmap_uint8, (224, 224), interpolation=cv2.INTER_CUBIC)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    image_cv2 = cv2.cvtColor(np.array(image_resized), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_cv2, 0.5, heatmap_color, 0.5, 0)
    return overlay, heatmap_norm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
model = model.float()
model.eval()

image_path = "./img_cat.jpg"
image = Image.open(image_path).convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]
grid_size = 16
num_patches = grid_size ** 2
hidden_dim = model.visual.transformer.width

with torch.no_grad():
    x = model.visual.conv1(image_input)
    x = x.reshape(1, hidden_dim, -1).permute(0, 2, 1)

    class_embedding = model.visual.class_embedding.to(x.dtype)
    class_token = class_embedding.expand(x.shape[0], 1, -1)
    x = torch.cat([class_token, x], dim=1)
    x = x + model.visual.positional_embedding.to(x.dtype)
    x = model.visual.ln_pre(x)

    mid_x = []
    for block in model.visual.transformer.resblocks:
        x = block(x)
        mid_x.append(x)

    f_feat = mid_x[-1][:, 0, :]
    f_feat = model.visual.ln_post(f_feat)
    if model.visual.proj is not None:
        f_feat = f_feat @ model.visual.proj

    for idx, x in enumerate(mid_x):
        x = model.visual.ln_post(x)

        if model.visual.proj is not None:
            x = x @ model.visual.proj

        p_feat = x[:, 1:, :]

        f_feat = f_feat / f_feat.norm(dim=-1, keepdim=True)
        p_feat = p_feat / p_feat.norm(dim=-1, keepdim=True)

        similarity = torch.einsum("ab,apb->ap", f_feat, p_feat).squeeze(0).cpu().numpy()
        heatmap = similarity.reshape(grid_size, grid_size)

        overlay_img, heatmap_norm = overlay_heatmap_on_image(image, heatmap)

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap="jet"),
            fraction=0.046, pad=0.04
        )
        plt.tight_layout()
        plt.savefig(f"./clip/clipvit_heat_cp/cat_block_{idx}.jpg", dpi=300)
        plt.close()
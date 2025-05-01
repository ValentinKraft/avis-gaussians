import numpy as np
import torch
from gaussian_me.model import GaussianModel

# Lade die .npz-Datei
path = "model/point_cloud/iteration_5000/point_cloud.npz"
data = np.load(path)

gaussians = GaussianModel(
    xyz=torch.tensor(data["xyz"]),
    sh_coefs=torch.tensor(data["features"]),
    scale=torch.tensor(data["scaling"]),
    rotation=torch.tensor(data["rotation"]),
    opacity=torch.tensor(data["opacity"]),
    sh_degree=3,  # Oder 0, falls du keine SHs verwendest
)

# Speichere als PLY
gaussians.save_ply("output_gaussians.ply")
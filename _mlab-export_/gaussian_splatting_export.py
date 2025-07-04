from mevis import *

import random
import os
import json
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

out_path = "C:/DEV/TESTS/gs/gaussian-splatting/_scene_"
transforms = {
    "camera_angle_x": 0.785398,
    "w": 1024,
    "h": 1024,
    "frames": []
}

def quaternion_from_matrix(matrix):
    return R.from_matrix(matrix[:3, :3]).as_quat()  # x, y, z, w
    
    
def access_image(node):
    image = ctx.field(node).image()
    if not image:
        print("Kein Bild gefunden.")
        return

    # Hole gesamten Bildbereich (Tile)
    tile = image.getTile((0, 0, 0, 0, 0, 0), image.imageExtent())
    if tile is None:
        print("Tile nicht gefunden.")
        return

    return image, tile
    
    
def generate_pca_covariances(voxel_coords, arr, spacing, neighborhood_size=3):
    covariances = []
    rotations = []
    scalings = []

    # Pad array to handle edge cases
    padded = np.pad(arr, neighborhood_size, mode='constant', constant_values=0)
    offset = neighborhood_size

    for coord in voxel_coords:
        z, y, x = coord
        # Local neighborhood window
        z0, z1 = z, z + 2 * neighborhood_size + 1
        y0, y1 = y, y + 2 * neighborhood_size + 1
        x0, x1 = x, x + 2 * neighborhood_size + 1

        local_patch = padded[z0:z1, y0:y1, x0:x1]
        # Get non-zero voxels as weighted points
        local_points = []
        for dz in range(local_patch.shape[0]):
            for dy in range(local_patch.shape[1]):
                for dx in range(local_patch.shape[2]):
                    val = local_patch[dz, dy, dx]
                    if val > 0:
                        # Voxel position in original array space
                        pz = (z + dz - neighborhood_size) * spacing[2]
                        py = (y + dy - neighborhood_size) * spacing[1]
                        px = (x + dx - neighborhood_size) * spacing[0]
                        local_points.append([px, py, pz])

        local_points = np.array(local_points)
        if local_points.shape[0] < 3:
            # Not enough points for PCA
            covariances.append(np.eye(3))
            rotations.append(np.eye(3))
            scalings.append(np.array([1, 1, 1]))
            continue

        # PCA: subtract mean, compute SVD
        mean = np.mean(local_points, axis=0)
        centered = local_points - mean
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        rotation = Vt.T  # columns are principal axes
        scaling = S / np.sum(S)  # normalize scaling for visualization

        # Covariance can be reconstructed as: rotation @ diag(scaling) @ rotation.T
        covariances.append(rotation @ np.diag(scaling**2) @ rotation.T)
        rotations.append(rotation)
        scalings.append(scaling)

    return covariances, rotations, scalings


def generate_random_splats(num_points=5000, output_path="points3D.txt"):
    
    output_path += "/sparse/0/points3D.txt"

    # points3D.txt schreiben
    with open(output_path, "w") as f:
        f.write("# 3D point list with one line per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        for i in range(num_points):
            wx = random.uniform(-0.2,0.2)
            wy = random.uniform(-0.2,0.2)
            wz = random.uniform(-0.2,0.2)

            r = random.randint(100, 255)
            g = random.randint(100, 255)
            b = random.randint(100, 255)

            f.write(f"{i} {wx:.6f} {wy:.6f} {wz:.6f} {r} {g} {b} 0.0 1 1 2 1\n")

    print(f"[✓] {num_points} Punkte aus Bild gespeichert nach: {output_path}")


def generate_weighted_splats_from_image(num_points=5000, output_path="points3D.txt"):
  
    output_path += "/sparse/0/points3D.txt"
    
    image, tile = access_image("Vesselness.output0")
    dicom_img, dicom_tile = access_image("SubImage.output0")

    # In NumPy-Array umwandeln
    arr = np.array(tile, dtype=np.float32)
    
    # Falls zusätzliche Dimensionen existieren, diese entfernen
    while arr.ndim > 3:
      arr = arr[0]

    # Bildgrößen und Weltinformationen
    dims = image.imageExtent()[1:4]  # (x,y,z)
    spacing = image.voxelSize()
    print(f"Spacing: {spacing}")
    m = image.voxelToWorldMatrix()#.getTranslation()
    origin = [m[0][3],m[1][3],m[2][3]]
    print(f"Origin: {origin}")

    # Wahrscheinlichkeitsverteilung erstellen (z. B. Werte > 0)
    arr_flat = arr.flatten()
    arr_flat[arr_flat < 0] = 0  # nur positive Werte erlauben
    total = np.sum(arr_flat)
    if total == 0:
        print("Alle Bildwerte sind 0.")
        return
    probs = arr_flat / total

    # Indexpositionen für Sampling
    all_indices = np.arange(len(arr_flat))

    # Ziehe Indizes gemäß Gewichtung
    chosen_indices = np.random.choice(all_indices, size=num_points, replace=False, p=probs)

    # Berechne Voxel-Koordinaten (z,y,x Reihenfolge beachten!)
    coords = np.unravel_index(chosen_indices, arr.shape)
    voxel_coords = np.stack(coords, axis=-1)  # (z, y, x)
    
    #covs, rots, scales = generate_pca_covariances(voxel_coords, arr, spacing)

    # points3D.txt schreiben
    with open(output_path, "w") as f:
        f.write("# 3D point list with one line per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        for i, (z, y, x) in enumerate(voxel_coords, 1):
            # Weltkoordinaten berechnen
            wx = (origin[0] + x * spacing[0] ) / 100.0
            wy = (origin[1] + y * spacing[1] ) / -100.0
            wz = (origin[2] + z * spacing[2] ) / -100.0

            r = dicom_tile[0,0,0,z,y,x]
            g = dicom_tile[0,0,0,z,y,x]
            b = dicom_tile[0,0,0,z,y,x]

            f.write(f"{i} {wx:.6f} {wy:.6f} {wz:.6f} {r} {g} {b} 0.0 1 1 2 1\n")

    print(f"[✓] {num_points} gewichtete Punkte aus Bild gespeichert nach: {output_path}")


def generate_weighted_splats_from_image_with_pca(num_points=5000, output_dir="output"):
    output_path = os.path.join(output_dir, "sparse/0")
    os.makedirs(output_path, exist_ok=True)

    image, tile = access_image("Vesselness.output0")
    dicom_img, dicom_tile = access_image("SubImage.output0")

    arr = np.array(tile, dtype=np.float32)
    while arr.ndim > 3:
        arr = arr[0]

    dims = image.imageExtent()[1:4]  # (x,y,z)
    spacing = image.voxelSize()
    m = image.voxelToWorldMatrix()
    origin = [m[0][3], m[1][3], m[2][3]]

    arr_flat = arr.flatten()
    arr_flat[arr_flat < 0] = 0
    total = np.sum(arr_flat)
    if total == 0:
        print("Alle Bildwerte sind 0.")
        return
    probs = arr_flat / total

    all_indices = np.arange(len(arr_flat))
    chosen_indices = np.random.choice(all_indices, size=num_points, replace=False, p=probs)
    coords = np.unravel_index(chosen_indices, arr.shape)
    voxel_coords = np.stack(coords, axis=-1)

    scalings = []
    rotations = []

    with open(os.path.join(output_path, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        for i, (z, y, x) in enumerate(voxel_coords, 1):
            wx = (origin[0] + x * spacing[0]) / 100.0
            wy = (origin[1] + y * spacing[1]) / -100.0
            wz = (origin[2] + z * spacing[2]) / -100.0

            r = dicom_tile[0, 0, 0, z, y, x]
            g = r
            b = r
            f.write(f"{i} {wx:.6f} {wy:.6f} {wz:.6f} {r} {g} {b} 0.0 1 1 2 1\n")

            # PCA für lokale Nachbarschaft
            window = 3
            zmin, zmax = max(0, z - window), min(arr.shape[0], z + window + 1)
            ymin, ymax = max(0, y - window), min(arr.shape[1], y + window + 1)
            xmin, xmax = max(0, x - window), min(arr.shape[2], x + window + 1)

            subvolume = arr[zmin:zmax, ymin:ymax, xmin:xmax]
            sub_coords = np.argwhere(subvolume > 0)
            if len(sub_coords) < 3:
                sub_coords = np.array([[0, 0, 0]])

            pca = PCA(n_components=3)
            pca.fit(sub_coords)

            scaling = pca.singular_values_ * np.mean(spacing) / 100.0
            rotation = pca.components_  # shape (3,3)

            scalings.append(scaling.tolist())
            rotations.append(rotation.tolist())

    # Speichern der zusätzlichen Splat-Informationen
    np.save(os.path.join(output_path, "scalings.npy"), np.array(scalings))
    np.save(os.path.join(output_path, "rotations.npy"), np.array(rotations))

    print(f"[✓] {num_points} gewichtete Punkte mit PCA gespeichert nach: {output_path}")



def render_images_and_generate_cameras_txt(num_imgs=100, output_path="", extent=100):
  
  image_width = transforms["w"]
  image_height = transforms["h"]
  focal_length = (image_width / 2) / np.tan(transforms["camera_angle_x"] / 2)
  camera_id = 1
  
  # render random cams and render images
  with open(os.path.join(output_path+"/sparse/0", "images.txt"), "w") as f:
    
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

    for i in range(num_imgs):
      
      radius = extent
      theta = random.uniform(0, 2 * np.pi)       # Azimut
      phi = random.uniform(0.2 * np.pi, 0.8 * np.pi)  # Polarwinkel, um nicht nur oben/unten zu landen

      x = radius * np.sin(phi) * np.cos(theta)
      y = radius * np.sin(phi) * np.sin(theta)
      z = radius * np.cos(phi)
      
      ctx.field("RotateAtTarget.inPosition").setValue([x,y,z])
      ctx.field("RotateAtTarget.update").touch()
      ctx.field("OffscreenRenderer.update").touch()
      ctx.field("ImageSave.filename").setValue(f"{output_path}/images/{i}.jpg")
      ctx.field("ImageSave.save").touch()

      rot = ctx.field("RotateAtTarget.outQuaternionRotation").value
      
      # Original Quaternion von Kamera zu Welt
      #q_wc = [rot[3], rot[0], rot[1], rot[2]]
      r_wc = R.from_quat([rot[0], -rot[1], -rot[2], rot[3]])  # [x, y, z, w]
      
      r_cw = r_wc.inv()
      
      qx, qy, qz, qw = r_cw.as_quat()

      # Kamerazentrum
      C = np.array([x, -y, -z])
      t = -r_cw.as_matrix() @ C
      
      f.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]/10.0} {t[1]/10.0} {t[2]/10.0} {camera_id} {i}.jpg\n")
  
  print(f"data exported to {output_path}")
  
##############################################

def update():
    
    #generate_weighted_splats_from_image(num_points=100000,output_path=out_path)
    generate_weighted_splats_from_image_with_pca(num_points=100000, output_dir=out_path)
    render_images_and_generate_cameras_txt(100,out_path,70)
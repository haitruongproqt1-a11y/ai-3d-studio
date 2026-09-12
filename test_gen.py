import sys, os, time
sys.path.append('TripoSR')
from PIL import Image
import numpy as np
import torch
import trimesh
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
from tsr.bake_texture import bake_texture
import xatlas

device = 'cuda:0'
print("Loading model...", flush=True)
t0 = time.time()
model = TSR.from_pretrained('stabilityai/TripoSR', config_name='config.yaml', weight_name='model.ckpt')
print(f"Model loaded in {time.time()-t0:.2f}s, sending to device...", flush=True)
model.renderer.set_chunk_size(8192)
model.to(device)

print("Preparing image...", flush=True)
img = Image.open('test_rembg.png')
img = resize_foreground(img, 0.85)
image_np = np.array(img).astype(np.float32) / 255.0
image_np = image_np[:, :, :3] * image_np[:, :, 3:4] + (1 - image_np[:, :, 3:4]) * 0.5
image = Image.fromarray((image_np * 255.0).astype(np.uint8))
image.save('input_processed_test.png')

t1 = time.time()
print("Inferring scene codes...", flush=True)
with torch.no_grad():
    scene_codes = model([image], device=device)
print(f"Scene codes done in {time.time()-t1:.2f}s", flush=True)

t2 = time.time()
print("Extracting mesh resolution 256...", flush=True)
meshes = model.extract_mesh(scene_codes, False, resolution=256)
print(f"Mesh extracted in {time.time()-t2:.2f}s, smoothing...", flush=True)
trimesh.smoothing.filter_taubin(meshes[0], lamb=0.5, nu=-0.53, iterations=10)

t3 = time.time()
print("Baking texture 1024...", flush=True)
bake_output = bake_texture(meshes[0], model, scene_codes[0], 1024)
print(f"Texture baked in {time.time()-t3:.2f}s, exporting obj...", flush=True)
xatlas.export('test_out.obj', meshes[0].vertices[bake_output['vmapping']], bake_output['indices'], bake_output['uvs'], meshes[0].vertex_normals[bake_output['vmapping']])

tex_img = Image.fromarray((bake_output['colors'] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)
tex_img.save('test_tex.png')

print("Exporting PBR GLB...", flush=True)
loaded = trimesh.load('test_out.obj')
mat = trimesh.visual.material.PBRMaterial(
    baseColorTexture=tex_img,
    roughnessFactor=0.35,
    metallicFactor=0.05
)
loaded.visual.material = mat
loaded.export('test_out.glb')
print(f"ALL DONE in {time.time()-t0:.2f}s! GLB size: {os.path.getsize('test_out.glb')} bytes", flush=True)

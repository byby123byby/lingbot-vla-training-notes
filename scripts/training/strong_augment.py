"""CPU image augmentation, replayed across LingBot's current/future views.

Public probabilities: configs/augmentation_recipe.json. Undisclosed strengths
are our explicit engineering defaults, not a reproduction of an author's code.
No model inference, generated image, task label or action modification.
"""
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageFilter


def sample_visual_augmentation_params(reference):
    if reference.ndim not in (3, 4) or reference.shape[-3] != 3:
        raise ValueError("Expected RGB CHW or TCHW")
    # Use the worker's torch RNG; do not create identical seeds in every worker.
    seed = int(torch.randint(0, 2**31 - 1, ()).item())
    rng = np.random.default_rng(seed)
    uniform = lambda a, b: float(rng.uniform(a, b))
    active = rng.random() >= .1
    return {
        "active": bool(active), "seed": seed,
        "color": bool(rng.random() < .8),
        "brightness": uniform(.25, 1.1), "contrast": uniform(.5, 1.5),
        "saturation": uniform(.4, 1.6),
        "geometry": bool(rng.random() < .5), "scale": uniform(.9, 1.1),
        "dx": uniform(-.06, .06), "dy": uniform(-.06, .06),
        "occlusion": bool(rng.random() < .95),
        "rectangles": [[uniform(0, 1), uniform(0, 1), uniform(.04, .12),
                        uniform(.04, .12), int(rng.integers(0, 3))]
                       for _ in range(int(rng.integers(1, 3)))],
        "noise": bool(rng.random() < .5), "noise_sigma": uniform(0, .025),
        "salt_pepper": bool(rng.random() < .15), "salt_density": uniform(.001, .005),
        "blur": bool(rng.random() < .25), "blur_sigma": uniform(.1, 1.0),
        "motion": bool(rng.random() < .10), "motion_axis": int(rng.integers(0, 2)),
        "motion_kernel": int(rng.choice([3, 5])),
        "jpeg": bool(rng.random() < .3), "jpeg_quality": int(rng.integers(35, 91)),
    }


def foreground_mask(rgb):
    """Conservative heuristic, NOT semantic segmentation.

    Protect the central 84% x 84%, high saturation and edge pixels, dilated.
    Only smooth peripheral background may be occluded; uncertain areas are kept.
    """
    h, w = rgb.shape[:2]
    saturation = rgb.max(-1) - rgb.min(-1)
    gray = rgb.mean(-1)
    edge = np.zeros((h, w), dtype=bool)
    edge[1:] |= np.abs(np.diff(gray, axis=0)) > .04
    edge[:, 1:] |= np.abs(np.diff(gray, axis=1)) > .04
    mask = edge | (saturation > .12)
    mask[int(h*.08):max(int(h*.92), 1), int(w*.08):max(int(w*.92), 1)] = True
    return np.asarray(Image.fromarray(mask.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(7))) > 0


def geometry(rgb, p):
    h, w = rgb.shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # Inverse sampling with replicate borders; bilinear interpolation.
    xx = np.clip((xx-(w-1)/2-p["dx"]*w)/p["scale"]+(w-1)/2, 0, w-1)
    yy = np.clip((yy-(h-1)/2-p["dy"]*h)/p["scale"]+(h-1)/2, 0, h-1)
    x0, y0 = xx.astype(int), yy.astype(int)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    wx, wy = (xx-x0)[..., None], (yy-y0)[..., None]
    return ((rgb[y0,x0]*(1-wx)+rgb[y0,x1]*wx)*(1-wy)
            +(rgb[y1,x0]*(1-wx)+rgb[y1,x1]*wx)*wy).astype(np.float32)


def apply_frame(rgb, p, protected=None):
    rng = np.random.default_rng(p["seed"])
    if p["geometry"]:
        rgb = geometry(rgb, p)
    if protected is None:
        protected = foreground_mask(rgb)
    if p["color"]:
        rgb = rgb*p["brightness"]
        mean = rgb.mean()
        rgb = (rgb-mean)*p["contrast"]+mean
        gray = rgb.mean(-1, keepdims=True)
        rgb = gray+(rgb-gray)*p["saturation"]
    rgb = np.clip(rgb, 0, 1)
    h, w = rgb.shape[:2]
    if p["occlusion"]:
        for x,y,rw,rh,kind in p["rectangles"]:
            x0,y0 = int(x*w),int(y*h)
            x1,y1 = min(w,x0+max(1,int(rw*w))),min(h,y0+max(1,int(rh*h)))
            if protected[y0:y1,x0:x1].any():
                continue
            shape = (y1-y0,x1-x0,3)
            if kind == 0:
                fill = rng.random(3)
            elif kind == 1:
                patch = Image.fromarray((rgb[y0:y1,x0:x1]*255).astype(np.uint8))
                fill = np.asarray(patch.resize((2,2)).resize((shape[1],shape[0]), Image.Resampling.NEAREST))/255
            else:
                fill = rng.random(shape)
            rgb[y0:y1,x0:x1] = fill
    if p["noise"]:
        rgb = rgb+rng.normal(0,p["noise_sigma"],rgb.shape)
    if p["salt_pepper"]:
        u = rng.random((h,w))
        rgb[u < p["salt_density"]/2] = 0
        rgb[u > 1-p["salt_density"]/2] = 1
    rgb = np.clip(rgb,0,1)
    if p["motion"]:
        k = p["motion_kernel"]
        pad = [(0,0),(0,0),(0,0)]; pad[p["motion_axis"]] = (k//2,k//2)
        padded = np.pad(rgb,pad,mode="edge")
        rgb = sum(np.take(padded,np.arange(rgb.shape[p["motion_axis"]])+i,axis=p["motion_axis"])
                  for i in range(k))/k
    if p["blur"] or p["jpeg"]:
        im = Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8))
        if p["blur"]:
            im = im.filter(ImageFilter.GaussianBlur(p["blur_sigma"]))
        if p["jpeg"]:
            buf = BytesIO(); im.save(buf,format="JPEG",quality=p["jpeg_quality"])
            buf.seek(0); im = Image.open(buf).convert("RGB")
        rgb = np.asarray(im,dtype=np.float32)/255
    return np.clip(rgb,0,1)


def apply_visual_augmentation(visual, params):
    if visual.ndim not in (3,4) or visual.shape[-3] != 3:
        raise ValueError("Expected RGB CHW or TCHW")
    if visual.device.type != "cpu":
        raise ValueError("Strong augmentation belongs in CPU data workers")
    if not params["active"]:
        return visual
    frames = visual.unsqueeze(0) if visual.ndim == 3 else visual
    byte_input = visual.dtype == torch.uint8
    if not byte_input and (not visual.is_floating_point() or visual.min()<0 or visual.max()>1):
        raise ValueError("Float images must be in [0,1]; byte images must be uint8")
    a = frames.detach().float().permute(0,2,3,1).numpy().copy()
    if byte_input:
        a /= 255
    # Protect the union over a video clip; no moving foreground is erased.
    mask = np.logical_or.reduce([foreground_mask(geometry(f,params) if params["geometry"] else f) for f in a])
    a = np.stack([apply_frame(f,params,mask) for f in a])
    if byte_input:
        a = np.rint(a*255).astype(np.uint8)
    out = torch.from_numpy(a).permute(0,3,1,2).to(visual.dtype)
    return out.squeeze(0) if visual.ndim == 3 else out

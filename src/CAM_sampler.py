import numpy as np
import torch
from torch import nn
from torchvision import transforms
from typing import Tuple

import cv2

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)

from elastic_vit_example.patch_sampler import PatchSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CAM_Sampler(PatchSampler):
    def __init__(self, model,threshold=150, patch_size=(16,16),
                 bigger_grid_patch_size=(32,32), seed=1000000009):
        super().__init__(patch_size, seed)
        self.bigger_grid_patch_size = bigger_grid_patch_size
        self.model = model
        self.threshold = threshold
        
        
        model.eval()
        for p in model.parameters():
            p.requires_grad = True  # needed to propagate gradients for grad-cam

        target_layers = [model.layer4]  # choose only last conv layer for the activation map
        self.cam = GradCAM(model=model, target_layers=target_layers)

    def __call__(self, img):
       return image_patching(img, model=self.model, cam=self.cam, patch_size=self.patch_size,
                                bigger_grid_patch_size=self.bigger_grid_patch_size,
                                threshold=self.threshold)
    

# every box has representation (x0,y0,x1,y1) where these are cords of upper left and lower right corners
def bounding_box_check(x,y, list_of_bounding_box_cords) -> bool:
  for box in list_of_bounding_box_cords:
    x0,y0,x1,y1 = box
    if (x >= x0 and x < x1) and (y >= y0 and y < y1):
      return False
  return True

# returns patches of a given size. If we want to return bigger patches, then we check if they collide with bounding boxes
def get_patches(image_tensor: torch.Tensor, x: int, y: int, w: int, h: int,  size: tuple=(16,16),
                mode="small", list_of_bounding_box_cords=[]) -> list:
  resize_transform = transforms.Resize((16,16),antialias=True)

  list_of_patches = []
  list_of_patch_cords = []
  for i in range(w // size[0]):
    for j in range(h // size[1]):
      x_temp, y_temp = x + i * size[0], y + j * size[1]
      w_temp, h_temp = size
      if mode == "small" or bounding_box_check(x_temp, y_temp, list_of_bounding_box_cords):
        patch = image_tensor[:, y_temp:y_temp+h_temp, x_temp:x_temp+w_temp]
        patch = resize_transform(patch)
        list_of_patches.append(patch)
        list_of_patch_cords.append((x_temp,y_temp, x_temp+w_temp,y_temp+h_temp))
  return list_of_patches



def image_patching(image_tensor: torch.Tensor, model: nn.Module, cam, patch_size: Tuple[int, int]=(8, 8), bigger_grid_patch_size: Tuple[int, int]=(32, 32),
                         threshold: int = 150, mode: str = 'val') -> Tuple:
    # # Convert the PyTorch tensor to a NumPy array and convert from CHW to HWC format
    # model.eval()
    # for p in model.parameters():
    #     p.requires_grad = True  # needed to propagate gradients for grad-cam

    # target_layers = [model.layer4]  # choose only last conv layer for the activation map

    rgb_img = (image_tensor / 255) #.squeeze(0)  FIXME
    rgb_img = np.transpose(rgb_img.numpy(), (1, 2, 0))

    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    label = torch.argmax(model(input_tensor.to(device)))

    # Construct the CAM object once, and then re-use it on many images:
    # cam = GradCAM(model=model, target_layers=target_layers) #, use_cuda=(device.type=='cuda'))
    cam.batch_size = 1

    output = model(input_tensor.to(device))
    pred_label = torch.argmax(output, dim=-1).item()

    if mode == 'train':
        targets = [ClassifierOutputTarget(label.item())]  # When we know the label --> during training
    else:
        targets = None  # When we don't have the label --> during validation and testing --> will take the label from argmax

    # Can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        aug_smooth=True)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]


    gray = (grayscale_cam * 255).astype("uint8")
    thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    if len(cnts) == 0:
      bigger_patches = get_patches(image_tensor, 0, 0, 224, 224, size=bigger_grid_patch_size)
      return torch.stack(bigger_patches)
    else:
      #bigger_patches = get_patches(image_tensor, 0, 0, 224,224, size=bigger_grid_patch_size)
      res = []
      list_of_bounding_box_cords = []

      for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if (w,h) < (16,16):
          continue

        if bigger_grid_patch_size[0] - x % bigger_grid_patch_size[0] >= x % bigger_grid_patch_size[0]:
                x = x - x % bigger_grid_patch_size[0]
        else:
            x = x + (bigger_grid_patch_size[0] - x % bigger_grid_patch_size[0])

        if bigger_grid_patch_size[1] - y % bigger_grid_patch_size[1] >= y % bigger_grid_patch_size[1]:
            y = y - y % bigger_grid_patch_size[1]
        else:
            y = y + (bigger_grid_patch_size[1] - y % bigger_grid_patch_size[1])


        if bigger_grid_patch_size[0] - w % bigger_grid_patch_size[0] >= w % bigger_grid_patch_size[0] and w - w % bigger_grid_patch_size[0] > 0:
            w = w - w % bigger_grid_patch_size[0]
        else:
            w = w + (bigger_grid_patch_size[0] - w % bigger_grid_patch_size[0])

        if bigger_grid_patch_size[1] - h % bigger_grid_patch_size[1] >= h % bigger_grid_patch_size[1] and  h - h % bigger_grid_patch_size[1] > 0:
            h = h - h % bigger_grid_patch_size[1]
        else:
            h = h + (bigger_grid_patch_size[1] - h % bigger_grid_patch_size[1])

        smaller_patches = get_patches(image_tensor,x,y,w,h,size=patch_size)
        res.append(smaller_patches)
        list_of_bounding_box_cords.append((x,y,x+w,y+h))

      bigger_patches = get_patches(image_tensor, 0, 0, 224,224, size=bigger_grid_patch_size, \
                                   mode="big", list_of_bounding_box_cords=list_of_bounding_box_cords)

      res.append(bigger_patches)
      res = sum(res, [])

      return torch.stack(res)
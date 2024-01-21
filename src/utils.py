import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple

def batch_histogram(data_tensor, num_classes=-1):
    """
    Computes histograms, even if in batches (as opposed to torch.histc and torch.histogram).
    Arguments:
        data_tensor: a D1 x ... x D_n torch.LongTensor
        num_classes (optional): the number of classes present in data.
                                If not provided, tensor.max() + 1 is used (an error is thrown if tensor is empty).
    Returns:
        A D1 x ... x D_{n-1} x num_classes 'result' torch.LongTensor,
        containing histograms of the last dimension D_n of tensor,
        that is, result[d_1,...,d_{n-1}, c] = number of times c appears in tensor[d_1,...,d_{n-1}].
    """
    maxd = data_tensor.max()
    nc = (maxd+1) if num_classes <= 0 else num_classes
    hist = torch.zeros((*data_tensor.shape[:-1], nc), dtype=data_tensor.dtype, device=data_tensor.device)
    ones = torch.tensor(1, dtype=hist.dtype, device=hist.device).expand(data_tensor.shape)
    hist.scatter_add_(-1, ((data_tensor * nc) // (maxd+1)).long(), ones)
    return hist


def draw_patches(Image: np.array, XY: List[Tuple[np.array, np.array]], color: str='lightgreen', linewidth: int = 1):
    """
    Plots patches over an Image, given the corner coordinates XY
    Args:
        - Image: a numpy array representing the image (e.g. of a flower)
        - XY: a list of tuples - with left upper corner and right lower corner coords per patch,
          e.g. XY = [(np.array([0, 0]), np.array([16, 16])), (np.array([16, 16]), np.array([64,64]))] <-- two patches
        - color: a string, color of the patches' circumference
        - linewidth: int, width of the lines creating each patch
    """
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    ax.imshow(Image)
    ax.set_title('Image patches')

    for coords in XY:
        left_corner, right_corner = coords[0], coords[1]
        radius = right_corner - left_corner  # get the rectangle's size
        rectangle = patches.Rectangle(left_corner, width=radius[0], height=radius[1], fill=False, color=color, linewidth=linewidth)
        ax.add_patch(rectangle)

    plt.axis('off')
    plt.show()
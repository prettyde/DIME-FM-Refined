import os
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class Voc2007Classification(torch.utils.data.Dataset):
    def __init__(self, data_root, image_set="train", transform=None):
        """
        Pascal voc2007 training/validation data: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        test data: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        """
        self.data_root = self._update_path(data_root, image_set)
        self.transfor
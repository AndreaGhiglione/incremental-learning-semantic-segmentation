import os.path as osp
import torch.utils.data as data

from PIL import Image

classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        is_aug (bool, optional): If you want to use the augmented train set or not (default is True)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self,
                 root,
                 image_set='train',
                 is_aug=True,
                 transform=None):

        self.root = osp.expanduser(root)
        self.year = "2012"

        self.transform = transform

        self.image_set = image_set
        base_dir = "PascalVOC12"
        voc_root = osp.join(self.root, base_dir)
        splits_dir = osp.join(voc_root, 'splits')

        if not osp.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use the script in data to download it')

        if is_aug and image_set == 'train':
            mask_dir = osp.join(voc_root, 'SegmentationClassAug')
            assert osp.exists(
                mask_dir), "SegmentationClassAug not found"
            split_f = osp.join(splits_dir, 'train_aug.txt')
        else:
            split_f = osp.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not osp.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        # remove leading \n
        with open(osp.join(split_f), "r") as f:
            file_names = [x[:-1].split(' ') for x in f.readlines()]

        # REMOVE FIRST SLASH OTHERWISE THE JOIN WILL start from root
        self.images = [(osp.join(voc_root, x[0][1:]), osp.join(voc_root, x[1][1:])) for x in file_names]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index][0]).convert('RGB')
        target = Image.open(self.images[index][1])
        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

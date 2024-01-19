from torch.utils.data import Dataset
from torchvision import transforms


class CarsDataset(Dataset):
    def __init__(self, base_dataset, split, sampler):
        self.base_dataset = base_dataset
        self.n_classes = 196
        self.sampler = sampler
        self.split = split
        self.augmentations = self._get_augmentations()

    def _get_augmentations(self):
        train_transforms = [
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-45, 45)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.Normalize(mean=[0.4746, 0.4638, 0.4588], std=[0.2941, 0.2942, 0.3022])
        ]
        test_transforms = [
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4746, 0.4638, 0.4588], std=[0.2941, 0.2942, 0.3022])
        ]

        if self.split == 'train':
            return transforms.Compose(train_transforms)
        else:
            return transforms.Compose(test_transforms)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        image = self.augmentations(image)
        x, coords = self.sampler(image)
        return x, coords, label, image
"""ImageFolder data handling for single-label remote-sensing scene classification."""

import random
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms


class SceneDataModule:
    """Build deterministic stratified train/validation splits and evaluation loaders."""

    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)

    def __init__(self, data_dir, imgsz=640, batch=32, workers=2, val_ratio=0.2, seed=42, sampling="balanced"):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.imgsz = int(imgsz)
        self.batch = int(batch)
        self.workers = int(workers)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.sampling = str(sampling)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Scene dataset directory not found: {self.data_dir}")

    @classmethod
    def build_transform(cls, imgsz, train=False):
        """Build the source-compatible preprocessing without requiring a dataset directory."""
        imgsz = int(imgsz)
        if imgsz <= 0:
            raise ValueError("imgsz must be positive.")
        operations = [transforms.Resize((imgsz, imgsz))]
        if train:
            operations.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                ]
            )
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=cls.NORMALIZE_MEAN, std=cls.NORMALIZE_STD),
            ]
        )
        return transforms.Compose(operations)

    def _transform(self, train=False):
        return self.build_transform(self.imgsz, train=train)

    @staticmethod
    def _image_folder(root, transform):
        dataset = datasets.ImageFolder(root, transform=transform)
        if not dataset.samples:
            raise ValueError(f"No class-folder images found under {root}.")
        return dataset

    def _stratified_indices(self, targets: Sequence[int], num_classes: int) -> Tuple[List[int], List[int]]:
        rng = random.Random(self.seed)
        train_indices, val_indices = [], []
        for class_id in range(num_classes):
            indices = [index for index, target in enumerate(targets) if int(target) == class_id]
            rng.shuffle(indices)
            val_count = max(1, int(len(indices) * self.val_ratio)) if len(indices) > 1 else 0
            val_indices.extend(indices[:val_count])
            train_indices.extend(indices[val_count:])
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        if not train_indices or not val_indices:
            raise ValueError("A non-empty stratified train/validation split could not be constructed.")
        return train_indices, val_indices

    @staticmethod
    def _balanced_sampler(targets: Sequence[int], indices: Sequence[int], num_classes: int):
        counts = [0] * num_classes
        for index in indices:
            counts[int(targets[index])] += 1
        class_weights = [len(indices) / (num_classes * max(count, 1)) for count in counts]
        sample_weights = [class_weights[int(targets[index])] for index in indices]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    def build_train_val(self):
        if self.batch < 2:
            raise ValueError("Training batch must be at least 2 because SceneClassifyHead uses BatchNorm1d.")
        train_root = self.data_dir / "train"
        val_root = self.data_dir / "val"
        pin_memory = torch.cuda.is_available()
        if train_root.is_dir() and val_root.is_dir():
            train_set = self._image_folder(train_root, self._transform(train=True))
            val_set = self._image_folder(val_root, self._transform(train=False))
            if train_set.classes != val_set.classes:
                raise ValueError("Train and validation class folders do not match.")
            train_indices = list(range(len(train_set)))
            sampler = (
                self._balanced_sampler(train_set.targets, train_indices, len(train_set.classes))
                if self.sampling == "balanced"
                else None
            )
            classes = train_set.classes
        else:
            train_full = self._image_folder(self.data_dir, self._transform(train=True))
            val_full = self._image_folder(self.data_dir, self._transform(train=False))
            train_indices, val_indices = self._stratified_indices(train_full.targets, len(train_full.classes))
            sampler = (
                self._balanced_sampler(train_full.targets, train_indices, len(train_full.classes))
                if self.sampling == "balanced"
                else None
            )
            train_set = Subset(train_full, train_indices)
            val_set = Subset(val_full, val_indices)
            classes = train_full.classes

        if len(train_set) < 2:
            raise ValueError("Scene classification requires at least two training samples.")
        drop_last = len(train_set) % self.batch == 1
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=self.workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader, list(classes)

    def build_eval(self, split="val"):
        split = str(split).lower()
        explicit_root = self.data_dir / split
        classes = None
        if split == "all":
            split_roots = [self.data_dir / name for name in ("train", "val", "test")]
            split_roots = [root for root in split_roots if root.is_dir()]
            if split_roots:
                split_datasets = [self._image_folder(root, self._transform(train=False)) for root in split_roots]
                classes = list(split_datasets[0].classes)
                if any(dataset.classes != classes for dataset in split_datasets[1:]):
                    raise ValueError("Explicit scene dataset splits do not share the same class folders.")
                dataset = ConcatDataset(split_datasets)
            else:
                dataset = self._image_folder(self.data_dir, self._transform(train=False))
        elif explicit_root.is_dir():
            dataset = self._image_folder(explicit_root, self._transform(train=False))
        else:
            full = self._image_folder(self.data_dir, self._transform(train=False))
            if split in {"train", "val"}:
                train_indices, val_indices = self._stratified_indices(full.targets, len(full.classes))
                dataset = Subset(full, train_indices if split == "train" else val_indices)
                dataset.classes = full.classes
            else:
                raise FileNotFoundError(f"No explicit '{split}' directory under {self.data_dir}.")
        if classes is None:
            classes = list(dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes)
        loader = DataLoader(
            dataset,
            batch_size=self.batch,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=torch.cuda.is_available(),
        )
        return loader, classes

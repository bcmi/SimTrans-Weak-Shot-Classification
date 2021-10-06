from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import DataLoader as Loader
from torchvision import transforms
from torch.utils.data.sampler import Sampler
import torch
import numpy as np
from PIL import Image


class DataHelper:
    def __init__(self, args):
        self.root_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        # source
        self.train_transforms = transforms.Compose(
            [transforms.RandomRotation(30), transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
             transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.test_transforms = transforms.Compose(
            [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.base_categories, self.novel_categories = [], []

    def __str__(self):
        return f'\n{len(self.base_categories)} Base categories:\n{self.base_categories[-20:]}\n' \
               f'\n{len(self.novel_categories)} Novel categories:\n{self.novel_categories[-20:]}\n'

    def get_base_and_novel_categories(self):
        return self.base_categories, self.novel_categories

    def get_novel_test_loader(self, shuffle=True):
        data_set = self._get_novel_test_set()
        return Loader(data_set, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def get_base_test_loader(self, shuffle=True):
        data_set = self._get_base_test_set()
        return Loader(data_set, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def get_clean_base_loader(self):
        data_set = self._get_clean_base_set()
        return Loader(data_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def get_noisy_novel_loader(self):
        data_set = self._get_noisy_novel_set()
        return Loader(data_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def get_clean_novel_loader(self):
        data_set = self._get_clean_novel_set()
        return Loader(data_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def get_noisy_base_loader(self):
        data_set = self._get_noisy_base_set()
        return Loader(data_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def _get_novel_test_set(self):
        raise NotImplementedError

    def _get_base_test_set(self):
        raise NotImplementedError

    def _get_clean_novel_set(self):
        raise NotImplementedError

    def _get_noisy_novel_set(self):
        raise NotImplementedError

    def _get_clean_base_set(self):
        raise NotImplementedError

    def _get_noisy_base_set(self):
        raise NotImplementedError

    def get_all_noisy_loader(self):
        set1 = self._get_noisy_base_set()
        set2 = self._get_noisy_novel_set()
        catted_set = cat_datasets(set1, set2)
        return Loader(catted_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def get_all_clean_loader(self):
        set1 = self._get_clean_base_set()
        set2 = self._get_clean_novel_set()
        catted_set = cat_datasets(set1, set2)
        return Loader(catted_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def get_generalized_loader(self):
        set1 = self._get_clean_base_set()
        set2 = self._get_noisy_novel_set()
        catted_set = cat_datasets(set1, set2)
        return Loader(catted_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def get_all_test_loader(self):
        set1 = self._get_base_test_set()
        set2 = self._get_novel_test_set()
        catted_set = cat_datasets(set1, set2)
        return Loader(catted_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def split_dataloader(data_loader, val_rate=0.2):
    train_set, val_set = data_loader.dataset.split(val_rate)
    return Loader(train_set, batch_size=data_loader.batch_size, shuffle=True, num_workers=data_loader.num_workers), \
           Loader(val_set, batch_size=data_loader.batch_size, shuffle=True, num_workers=data_loader.num_workers),


def cat_datasets(set1, set2):
    catted_set = DataSet(set1.root_path, set1.categories + set2.categories, set1.transform)
    catted_set.image_list = []
    for p, c in set1.image_list:
        catted_set.image_list.append((p, catted_set.category2int[set1.int2category[c]]))
    for p, c in set2.image_list:
        catted_set.image_list.append((p, catted_set.category2int[set2.int2category[c]]))

    return catted_set


class DataSet:
    def __init__(self, root_path, categories, transform):
        self.root_path = root_path
        self.categories = categories
        self.transform = transform
        self.int2category = dict(zip(range(len(self.categories)), self.categories))
        self.category2int = {v: k for k, v in enumerate(self.categories)}

    def num_classes(self):
        return len(self.categories)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_path, category = self.image_list[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, category, img_path

    def split(self, rate=0.2):
        train_list, val_list = [], []
        image_dict = {}
        for k in self.int2category.keys():
            image_dict[k] = []
        for p, c in self.image_list:
            image_dict[c].append(p)

        for k, v in image_dict.items():
            val_length = int(len(v) * rate)
            train_list += [(p, k) for p in v[val_length:]]
            val_list += [(p, k) for p in v[:val_length]]

        train_set = DataSet(self.root_path, self.categories, self.transform)
        val_set = DataSet(self.root_path, self.categories, self.transform)
        train_set.image_list = train_list
        val_set.image_list = val_list
        return train_set, val_set

    def get_dict(self):
        data_dict = {}
        for p, c in self.image_list:
            category = self.int2category[c]
            if category in data_dict.keys():
                data_dict[category].append(p)
            else:
                data_dict[category] = [p]

        return data_dict

    def get_category_length(self):
        data_dict = self.get_dict()
        return {k: len(v) for k, v in data_dict.items()}

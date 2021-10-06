from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader as Loader

import numpy as np

def get_train_loader(args, data_loader):
    return Loader(data_loader.dataset,
                  batch_sampler=RandomSampler(data_loader.dataset, args.batch_size, args.epoch_iter),
                  batch_size=0, shuffle=None,
                  drop_last=None, sampler=None,
                  num_workers=args.num_workers)


class RandomSampler(Sampler):
    def __init__(self, dataset, batch_size, epoch_iter):
        self.batch_size = batch_size
        image_dict = {}
        for k in dataset.int2category.keys():
            image_dict[k] = []
        for dataset_idx, (p, c) in enumerate(dataset.image_list):
            image_dict[c].append(dataset_idx)

        class_keys = list(image_dict.keys())

        self.batch_list = []
        for i in range(epoch_iter):
            batch = []
            for j in range(self.batch_size):
                selected_c = int(np.random.choice(class_keys))
                batch += np.random.choice(image_dict[selected_c], 1, replace=False).tolist()

            self.batch_list.append(batch)

        return

    def __iter__(self):
        return iter(self.batch_list)

    def __len__(self):
        return len(self.batch_list)

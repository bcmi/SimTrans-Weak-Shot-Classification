import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import shutil
import os
import numpy as np


def vis_acc(ys, ls, fpath=None):
    cs = ['blue', 'red', 'y', 'black']
    x = [i for i in range(len(ys[0]))]
    fig = plt.figure(figsize=(10, 10), dpi=200)
    for i, (y, l) in enumerate(zip(ys, ls)):
        plt.plot(x, y, label=l, linewidth=3, c=cs[i])

    plt.ylim(0, 1)
    plt.legend()
    if fpath is None:
        plt.show()
    else:
        plt.savefig(fpath)
        plt.close()
    return

def vis_cluster(image_list, cluster_list, show_categories, save_path=None):
    def mv(source, target):
        os.makedirs(target, exist_ok=True)
        shutil.copy(source, target)

    for image_path, cluster in zip(image_list, cluster_list):
        category = image_path.split(os.path.sep)[-2]
        source_image_path = f'{image_path}'
        target_image_path = f'{save_path}/{category}/{cluster}'
        mv(source_image_path, target_image_path)
    return


def vis_density(image_list, distances,
                show_num=1, save_path=None):
    def save_image_pair(p1, p2, v, save_path=None):
        im1 = plt.imread(p1)
        im2 = plt.imread(p2)

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(im1)

        ax2 = fig.add_subplot(122)
        ax2.imshow(im2)
        plt.title(f'{p1}\n'
                  f'{p2}\n'
                  f'Distance: {v:.4}')
        plt.savefig(save_path)
        plt.close()
        return

    def show_density(density, image_path, fpath=None):
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.title(f'Density: {density}')
        plt.savefig(fpath)
        return

    for c, (ids, dist, density) in enumerate(distances[:show_num]):
        image_path = np.array(image_list)[ids]

        sorted_idx = np.argsort(density)
        for rank, idx in enumerate(sorted_idx):
            show_density(density[idx], image_path[idx],
                         f'{save_path}/train_viz/{c}-{rank}.jpg')
        # vis distances
        # dist_value = []
        # dist_pair = []
        # for i in range(dist.shape[0]):
        #     for j in range(i + 1, dist.shape[1]):
        #         dist_value.append(dist[i, j])
        #         dist_pair.append((i, j))
        #
        # sorted_idx = np.argsort(dist_value)
        # for rank, idx in enumerate(sorted_idx):
        #     i, j = dist_pair[idx]
        #
        #     save_image_pair(image_path[i], image_path[j], dist_value[idx],
        #                     f'{save_path}/train_viz/{c}-{rank}.jpg')

    return


if __name__ == '__main__':
    ys = [[i / 2 + (np.random.rand() - 0.5) * 10 for i in range(150)],
          [i / 2 + (np.random.rand()) * i / 10 for i in range(150)]]
    ls = ['train', 'test']
    vis_acc(ys, ls, 'pretrained/t.png')
    d = 1

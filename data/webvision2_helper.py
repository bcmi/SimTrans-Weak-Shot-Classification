from data.helper import *
import os
import scipy.io as sio
import numpy as np

random_permutation = [481, 671, 877, 844, 436, 122, 75, 101, 662, 473, 563, 441, 985, 549, 791, 969, 606, 9, 663, 422,
                      111, 85, 84, 693, 908, 822, 666, 297, 311, 706, 355, 793, 981, 619, 768, 451, 333, 442, 404, 158,
                      255, 795, 405, 82, 465, 133, 62, 886, 321, 880, 519, 734, 570, 837, 203, 299, 715, 429, 721, 44,
                      369, 860, 421, 991, 823, 613, 700, 268, 567, 105, 926, 265, 412, 925, 10, 975, 169, 977, 676, 416,
                      118, 716, 56, 254, 548, 557, 523, 386, 609, 145, 362, 741, 94, 506, 638, 565, 153, 90, 537, 34,
                      45, 28, 859, 430, 207, 474, 248, 851, 849, 77, 674, 543, 11, 3, 181, 256, 273, 731, 511, 494, 782,
                      339, 655, 63, 884, 349, 695, 100, 387, 912, 819, 688, 337, 428, 334, 88, 159, 173, 279, 218, 587,
                      26, 527, 502, 560, 14, 996, 826, 806, 78, 869, 530, 108, 956, 812, 931, 407, 246, 626, 42, 487,
                      784, 847, 217, 272, 939, 245, 471, 899, 87, 196, 200, 766, 95, 116, 986, 49, 168, 705, 226, 897,
                      257, 382, 401, 48, 484, 833, 380, 296, 498, 74, 208, 496, 338, 96, 262, 933, 551, 737, 25, 559,
                      640, 456, 507, 744, 277, 278, 578, 38, 915, 786, 727, 24, 470, 201, 415, 593, 163, 493, 504, 435,
                      162, 172, 544, 252, 491, 287, 316, 237, 185, 865, 746, 541, 827, 625, 283, 60, 221, 879, 536, 81,
                      13, 152, 249, 426, 863, 917, 595, 505, 166, 629, 346, 861, 176, 521, 353, 417, 450, 406, 98, 872,
                      188, 468, 199, 864, 304, 953, 762, 177, 532, 331, 348, 194, 855, 66, 603, 22, 377, 947, 93, 155,
                      623, 750, 460, 195, 888, 892, 555, 320, 434, 788, 350, 35, 584, 300, 91, 413, 516, 79, 76, 53, 12,
                      659, 103, 873, 227, 359, 590, 47, 959, 958, 824, 509, 728, 598, 697, 282, 627, 120, 673, 314, 295,
                      290, 191, 628, 510, 137, 134, 936, 805, 738, 810, 739, 259, 882, 433, 890, 957, 572, 894, 966,
                      459, 774, 732, 18, 236, 874, 769, 612, 478, 390, 748, 512, 43, 520, 138, 336, 935, 113, 664, 669,
                      914, 140, 219, 878, 161, 696, 974, 968, 909, 709, 216, 887, 370, 86, 142, 690, 293, 820, 211, 868,
                      689, 146, 704, 220, 212, 821, 341, 798, 929, 875, 561, 617, 817, 242, 681, 214, 714, 318, 154,
                      635, 357, 83, 489, 699, 949, 46, 667, 940, 399, 2, 571, 967, 397, 241, 761, 911, 672, 658, 445,
                      71, 132, 992, 57, 239, 274, 938, 482, 340, 374, 979, 114, 816, 325, 457, 607, 707, 157, 924, 475,
                      476, 983, 110, 916, 240, 657, 112, 661, 393, 209, 400, 31, 634, 545, 550, 317, 852, 854, 574, 285,
                      963, 501, 621, 643, 260, 458, 759, 384, 764, 141, 948, 107, 665, 955, 698, 978, 307, 987, 684,
                      757, 825, 485, 710, 29, 647, 213, 354, 836, 364, 402, 930, 175, 440, 437, 409, 130, 179, 984, 722,
                      251, 809, 718, 592, 650, 529, 796, 379, 910, 997, 449, 150, 944, 266, 980, 641, 431, 360, 229,
                      719, 646, 853, 233, 589, 463, 787, 64, 17, 785, 881, 866, 711, 4, 499, 703, 558, 585, 815, 8, 856,
                      999, 503, 616, 469, 648, 862, 733, 747, 222, 51, 61, 358, 682, 736, 838, 310, 486, 743, 998, 244,
                      432, 932, 50, 395, 455, 660, 301, 789, 539, 775, 687, 305, 995, 831, 900, 167, 735, 37, 403, 376,
                      632, 148, 919, 790, 614, 842, 326, 182, 776, 547, 518, 653, 119, 267, 323, 927, 990, 156, 144,
                      204, 385, 27, 69, 965, 180, 677, 448, 495, 294, 500, 586, 398, 922, 366, 143, 89, 446, 371, 546,
                      577, 54, 391, 345, 533, 579, 644, 1, 492, 840, 186, 170, 513, 834, 223, 770, 556, 464, 773, 652,
                      800, 928, 52, 946, 187, 540, 151, 760, 923, 99, 70, 365, 675, 308, 192, 439, 645, 850, 183, 234,
                      604, 23, 20, 275, 361, 466, 857, 937, 184, 524, 528, 461, 758, 982, 717, 235, 813, 811, 127, 906,
                      678, 59, 423, 303, 918, 261, 971, 367, 420, 964, 373, 808, 713, 472, 723, 755, 197, 867, 783, 480,
                      656, 269, 389, 526, 802, 870, 128, 780, 883, 292, 131, 905, 467, 745, 67, 102, 702, 620, 136, 309,
                      109, 871, 281, 15, 284, 564, 65, 651, 497, 973, 576, 765, 189, 583, 972, 961, 408, 202, 372, 73,
                      347, 414, 32, 843, 807, 313, 708, 901, 291, 243, 692, 490, 215, 597, 902, 36, 418, 891, 582, 580,
                      895, 411, 797, 263, 123, 462, 388, 618, 752, 670, 691, 680, 941, 649, 381, 117, 363, 5, 781, 636,
                      224, 298, 630, 954, 749, 396, 633, 174, 522, 730, 639, 190, 104, 763, 903, 605, 225, 288, 624,
                      238, 16, 535, 637, 356, 125, 942, 712, 542, 508, 801, 841, 896, 0, 729, 206, 751, 447, 58, 425,
                      324, 566, 343, 55, 694, 383, 19, 264, 803, 232, 814, 315, 124, 921, 562, 830, 989, 247, 352, 596,
                      845, 799, 952, 160, 876, 683, 368, 72, 276, 375, 943, 302, 951, 335, 885, 777, 126, 394, 97, 993,
                      258, 839, 286, 720, 271, 960, 329, 832, 525, 164, 410, 139, 740, 438, 135, 41, 115, 250, 615, 767,
                      573, 178, 588, 452, 742, 92, 568, 554, 608, 231, 601, 39, 228, 198, 668, 594, 794, 602, 7, 753,
                      754, 701, 30, 829, 622, 319, 230, 531, 351, 835, 453, 344, 443, 147, 306, 171, 477, 312, 913, 534,
                      392, 424, 165, 654, 907, 599, 804, 253, 945, 483, 193, 33, 818, 893, 591, 538, 642, 479, 68, 80,
                      581, 210, 444, 771, 569, 342, 40, 517, 322, 427, 129, 575, 772, 488, 962, 970, 994, 610, 332, 270,
                      149, 756, 848, 858, 514, 330, 988, 280, 21, 934, 419, 898, 950, 327, 846, 552, 904, 724, 779, 289,
                      454, 378, 726, 792, 6, 631, 106, 920, 121, 778, 328, 205, 725, 685, 553, 889, 600, 515, 611, 828,
                      686, 679, 976]


def _get_all_categories(fpath):
    classes = []
    for line in open(fpath).readlines():
        classes.append(line[:9])
    return classes


class WebVisionHelper(DataHelper):
    def __init__(self, args):
        super(WebVisionHelper, self).__init__(args)
        self.name = 'WebVision'
        self.clean_path = args.data_path.replace("WebVision2", "ImageNet/train")
        self.all_categories = sorted(os.listdir(f'{self.clean_path}'))

        if args.novel_split in [-1, 118]:
            args.novel_split = 118
        else:
            raise NotImplementedError(f'Split {args.novel_split} not support yet.')

        for i, category in enumerate(self.all_categories):
            if i < args.novel_split:
                self.novel_categories.append(category)
            else:
                self.base_categories.append(category)

        return

    def _get_novel_test_set(self):
        return SourceTestSet(self.root_path, self.novel_categories, self.test_transforms)

    def _get_base_test_set(self):
        return SourceTestSet(self.root_path, self.base_categories, self.test_transforms)

    def _get_clean_base_set(self):
        return SourceTrainSet(self.root_path, self.base_categories, self.train_transforms)

    def _get_noisy_novel_set(self):
        return WebSet(self.root_path, self.novel_categories, self.train_transforms)


class SourceTrainSet(DataSet):
    def __init__(self, root_path, categories, transform=None, max_num=1000):
        super(SourceTrainSet, self).__init__(root_path, categories, transform)
        ImageNet_path = self.root_path.replace("WebVision2", "ImageNet")

        all_train_files = f'{ImageNet_path}/train_files.txt'
        self.image_list = []

        for line in open(all_train_files).readlines():
            image_path, WNID = line.strip().split(',')
            if WNID in categories:
                full_image_path = f'{ImageNet_path}/{image_path}'
                assert os.path.exists(full_image_path), f'Not found: {full_image_path}'
                self.image_list.append((full_image_path,
                                        self.category2int[WNID]))
        return


class SourceTestSet(DataSet):
    def __init__(self, root_path, categories, transform=None):
        super(SourceTestSet, self).__init__(root_path, categories, transform)
        ImageNet_path = self.root_path.replace("WebVision2", "ImageNet")
        label_path = 'ILSVRC2012_validation_ground_truth.txt'
        labels = open(f'{ImageNet_path}/{label_path}').readlines()

        from scipy.io import loadmat
        mats = loadmat(f'{ImageNet_path}/meta.mat')['synsets']

        to_WNID = {}
        for line in mats:
            info = line[0].base.tolist()[0]
            ILSVRC2012_ID = info[0][0, 0]
            WNID = info[1][0]
            to_WNID[ILSVRC2012_ID] = WNID

        self.image_list = []
        for line in sorted(os.listdir(f'{ImageNet_path}/val')):
            idx = int(line.split('_')[-1].split('.')[0]) - 1
            ILSVRC2012_ID = int(labels[idx].strip())
            WNID = to_WNID[ILSVRC2012_ID]

            if WNID in categories:
                self.image_list.append((f'{ImageNet_path}/val/{line}',
                                        self.category2int[WNID]))
        return


class WebSet(DataSet):

    def __init__(self, root_path, categories, transform=None, max_noisy_images_per=None):
        super(WebSet, self).__init__(root_path, categories, transform)
        all_train_files = f'{root_path}/info/train_filelist_all.txt'
        train_files_118 = f'{root_path}/train_files_118.txt'

        queries_synsets_map = f'{root_path}/info/queries_synsets_map.txt'
        synsets = f'{root_path}/info/synsets.txt'

        query_to_synset = {}
        synset_to_WNID = {}

        for line in open(queries_synsets_map).readlines():
            query, synset = line.strip().split(' ')
            query_to_synset[int(query)] = int(synset)

        for i, line in enumerate(open(synsets).readlines()):
            synset_to_WNID[i] = line[:9]

        self.image_list = []

        for line in open(train_files_118).readlines():
            image_path, synset = line.strip().split(' ')

            WNID = synset_to_WNID[int(synset)]
            if WNID in categories:
                full_image_path = f'{root_path}/{image_path}'
                full_image_path = full_image_path.replace('WebVision2', 'WebVision2/subset118')
                assert os.path.exists(full_image_path), f'Not found: {full_image_path}'
                self.image_list.append((full_image_path,
                                        self.category2int[WNID]))
        return


def __unzip_ImageNet(root_path):
    tars = sorted(os.listdir(root_path))
    for tar in tars:
        name = tar.split('.')[0]
        os.system(f'mkdir {root_path}/{name}')
        os.system(f'tar -xvf {root_path}/{name}.tar -C {root_path}/{name}')
        tar

    return


def __extract_WebVision(dataset):
    import shutil
    new_dir = 'subset118'
    for image_path, label in dataset.image_list:
        new_dir_path = os.path.dirname(image_path).replace('WebVision2', new_dir)
        os.makedirs(new_dir_path, exist_ok=True)
        new_image_path = image_path.replace('WebVision2', new_dir)
        shutil.copy(image_path, new_image_path)

    return


def __generate_file_list_for_ImageNet(dataset):
    f = open('../dataset/ImageNet/train_files.txt', 'a')

    for (image_path, label) in dataset.image_list:
        WNID = dataset.int2category[label]
        line = f"{image_path.split('ImageNet/')[1]},{WNID}\n"
        f.write(line)
    f.close()
    return


def __generate_file_list_for_WebVision118(dataset):
    f = open('../dataset/ImageNet/train_files_118.txt', 'a')

    for (image_path, label) in dataset.image_list:
        WNID = dataset.int2category[label]
        line = f"{image_path.split('ImageNet/')[1]},{WNID}\n"
        f.write(line)
    f.close()
    return


if __name__ == '__main__':
    import easydict
    from tqdm import tqdm

    root_path = '../../dataset/WebVision2'
    # __unzip_ImageNet(root_path)

    args = easydict.EasyDict()
    args.data_path = root_path
    args.batch_size = 8
    args.num_workers = 0
    args.novel_split = 118
    wv = WebVisionHelper(args)
    # a = wv._get_novel_test_set()
    # b = wv._get_base_test_set()

    # tmp = SourceTrainSet(wv.root_path, wv.all_categories, wv.train_transforms)
    # __generate_file_list_for_ImageNet(tmp)
    # c = wv._get_clean_base_set()
    d = wv._get_noisy_novel_set()
    # __extract_WebVision(d)
    root_path

    'data/meta/WebVision2/ILSVRC2012_validation_ground_truth.txt'
    queries = '../dataset/WebVision2/info/queries.txt'
    queries_synsets_map = '../dataset/WebVision2/info/queries_synsets_map.txt'
    synsets = '../dataset/WebVision2/info/synsets.txt'
    train_filelist_all = '../dataset/WebVision2/info/train_filelist_all.txt'

    root_path

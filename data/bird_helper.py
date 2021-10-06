from data.helper import *
import os
import scipy.io as sio
import numpy as np


def read_split_from_file(fpath):
    classes = []
    for line in open(fpath).readlines():
        classes.append(line.strip())
    return classes


class CUBHelper(DataHelper):
    # the split is followed at https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly
    def __init__(self, args):
        super(CUBHelper, self).__init__(args)
        self.name = 'Bird'
        self.base_categories = read_split_from_file('data/meta/CUB/trainvalclasses.txt')
        self.novel_categories = read_split_from_file('data/meta/CUB/testclasses.txt')
        return

    def _get_novel_test_set(self):
        return SourceTestSet(self.root_path, self.novel_categories, self.test_transforms)

    def _get_base_test_set(self):
        return SourceTestSet(self.root_path, self.base_categories, self.test_transforms)

    def _get_clean_novel_set(self):
        return SourceTrainSet(self.root_path, self.novel_categories, self.train_transforms)

    def _get_clean_base_set(self):
        return SourceTrainSet(self.root_path, self.base_categories, self.train_transforms)

    def _get_noisy_novel_set(self):
        return WebSet(self.root_path, self.novel_categories, self.train_transforms)

    def _get_noisy_base_set(self):
        return NoisySet(self.root_path, self.base_categories, self.train_transforms)

    def _get_synthetic_set(self):
        return SyntheticSet(self.root_path, self.novel_categories[:-2], self.train_transforms)

    def _get_synthetic_test_set(self):
        return SourceTestSet(self.root_path, self.novel_categories[:-2], self.test_transforms)


class SyntheticSet(DataSet):
    def __init__(self, root_path, categories, transform=None):
        super(SyntheticSet, self).__init__(root_path, categories, transform)
        self.image_list = []
        synthetic_dict = torch.load('data/meta/synthetic_dict.pth')
        assert len(synthetic_dict) == len(categories)
        for category, paths in synthetic_dict.items():
            assert category in self.categories
            self.image_list += [(path.replace('\\', os.sep), self.category2int[category]) for path in paths]
        return


class SourceTrainSet(DataSet):
    def __init__(self, root_path, categories, transform=None, max_num=1000):
        super(SourceTrainSet, self).__init__(root_path, categories, transform)

        train_ids = []
        for line in open(os.path.join(root_path, 'CUB_200_2011', 'train_test_split.txt')).readlines():
            img_id, is_train = line.strip().split(' ')
            if is_train == '1':
                train_ids.append(img_id)

        category_count = {}
        for cname in self.categories:
            category_count[cname] = 0

        self.image_list = []
        for line in open(os.path.join(root_path, 'CUB_200_2011', 'images.txt')).readlines():
            img_id, img_path = line.strip().split(' ')
            category, img_name = img_path.split('/')
            if img_id in train_ids:
                if category in categories:
                    if category_count[category] >= max_num:
                        continue
                    self.image_list.append((os.path.join(root_path, 'CUB_200_2011', 'images', img_path),
                                            self.category2int[category]))
                    category_count[category] += 1
        return


class SourceTestSet(DataSet):
    def __init__(self, root_path, categories, transform=None):
        super(SourceTestSet, self).__init__(root_path, categories, transform)

        train_ids = []
        for line in open(os.path.join(root_path, 'CUB_200_2011', 'train_test_split.txt')).readlines():
            img_id, is_train = line.strip().split(' ')
            if is_train == '0':
                train_ids.append(img_id)

        self.image_list = []
        for line in open(os.path.join(root_path, 'CUB_200_2011', 'images.txt')).readlines():
            img_id, img_path = line.strip().split(' ')
            category, img_name = img_path.split('/')
            if img_id in train_ids:
                if category in categories:
                    self.image_list.append((os.path.join(root_path, 'CUB_200_2011', 'images', img_path),
                                            self.category2int[category]))
        return


# class NoisySet(DataSet):
#     def _get_web_name(self, category_name):
#         return (category_name.split('.')[1] + '_bird').replace('_', '+')
#
#     def __init__(self, root_path, categories, transform=None, max_noisy_images_per=None):
#         super(NoisySet, self).__init__(root_path, categories, transform)
#
#         self.image_list = []
#         for category_name in categories:
#
#             web_name = self._get_web_name(category_name)
#             dir_path = os.path.join(self.root_path, 'CUB_Google_200class', web_name)
#             if not os.path.exists(dir_path):
#                 print(category_name)
#                 print(dir_path)
#                 raise FileNotFoundError
#
#             image_names = sorted(os.listdir(dir_path))
#             category_list = [(os.path.join(dir_path, image_name),
#                               self.category2int[category_name]) for
#                              image_name in image_names]
#
#             self.image_list += category_list[:max_noisy_images_per]
#
#         return


class WebSet(DataSet):

    def __init__(self, root_path, categories, transform=None, max_noisy_images_per=None):
        super(WebSet, self).__init__(root_path, categories, transform)

        self.image_list = []
        for category_name in categories:

            web_name = category_name
            dir_path = os.path.join(self.root_path, 'CUB_web', web_name)
            if not os.path.exists(dir_path):
                print(category_name)
                print(dir_path)
                raise FileNotFoundError

            image_names = sorted(os.listdir(dir_path))
            category_list = [(os.path.join(dir_path, image_name),
                              self.category2int[category_name]) for
                             image_name in image_names]

            self.image_list += category_list[:max_noisy_images_per]
        print(max_noisy_images_per)
        return


'''
===========================================
The Caltech-UCSD Birds-200-2011 Dataset
===========================================

For more information about the dataset, visit the project website:

  http://www.vision.caltech.edu/visipedia

If you use the dataset in a publication, please cite the dataset in
the style described on the dataset website (see url above).

Directory Information
---------------------

- images/
    The images organized in subdirectories based on species. See 
    IMAGES AND CLASS LABELS section below for more info.
- parts/
    15 part locations per image. See PART LOCATIONS section below 
    for more info.
- attributes/
    322 binary attribute labels from MTurk workers. See ATTRIBUTE LABELS 
    section below for more info.



=========================
IMAGES AND CLASS LABELS:
=========================
Images are contained in the directory images/, with 200 subdirectories (one for each bird species)

------- List of image files (images.txt) ------
The list of image file names is contained in the file images.txt, with each line corresponding to one image:

<image_id> <image_name>
------------------------------------------


------- Train/test split (train_test_split.txt) ------
The suggested train/test split is contained in the file train_test_split.txt, with each line corresponding to one image:

<image_id> <is_training_image>

where <image_id> corresponds to the ID in images.txt, and a value of 1 or 0 for <is_training_image> denotes that the file is in the training or test set, respectively.
------------------------------------------------------


------- List of class names (classes.txt) ------
The list of class names (bird species) is contained in the file classes.txt, with each line corresponding to one class:

<class_id> <class_name>
--------------------------------------------


------- Image class labels (image_class_labels.txt) ------
The ground truth class labels (bird species labels) for each image are contained in the file image_class_labels.txt, with each line corresponding to one image:

<image_id> <class_id>

where <image_id> and <class_id> correspond to the IDs in images.txt and classes.txt, respectively.
---------------------------------------------------------





=========================
BOUNDING BOXES:
=========================

Each image contains a single bounding box label.  Bounding box labels are contained in the file bounding_boxes.txt, with each line corresponding to one image:

<image_id> <x> <y> <width> <height>

where <image_id> corresponds to the ID in images.txt, and <x>, <y>, <width>, and <height> are all measured in pixels




=========================
PART LOCATIONS:
=========================

------- List of part names (parts/parts.txt) ------
The list of all part names is contained in the file parts/parts.txt, with each line corresponding to one part:

<part_id> <part_name>
------------------------------------------


------- Part locations (parts/part_locs.txt) ------
The set of all ground truth part locations is contained in the file parts/part_locs.txt, with each line corresponding to the annotation of a particular part in a particular image:

<image_id> <part_id> <x> <y> <visible>

where <image_id> and <part_id> correspond to the IDs in images.txt and parts/parts.txt, respectively.  <x> and <y> denote the pixel location of the center of the part.  <visible> is 0 if the part is not visible in the image and 1 otherwise.
----------------------------------------------------------


------- MTurk part locations (parts/part_click_locs.txt) ------
A set of multiple part locations for each image and part, as perceived by multiple MTurk users is contained in parts/part_click_locs.txt, with each line corresponding to the annotation of a particular part in a particular image by a different MTurk worker:

<image_id> <part_id> <x> <y> <visible> <time>

where <image_id>, <part_id>, <x>, <y> are in the same format as defined in parts/part_locs.txt, and <time> is the time in seconds spent by the MTurk worker.
----------------------------------------------------------



=========================
ATTRIBUTE LABELS:
=========================

------- List of attribute names (attributes/attributes.txt) ------
The list of all attribute names is contained in the file attributes/attributes.txt, with each line corresponding to one attribute:

<attribute_id> <attribute_name>
------------------------------------------------------------------


------- List of certainty names (attributes/certainties.txt) ------
The list of all certainty names (used by workers to specify their certainty of an attribute response of is contained in the file attributes/certainties.txt, with each line corresponding to one certainty:

<certainty_id> <certainty_name>
-------------------------------------------------------------------


------- MTurk image attribute labels (attributes/image_attribute_labels.txt) ------
The set of attribute labels as perceived by MTurkers for each image is contained in the file attributes/image_attribute_labels.txt, with each line corresponding to one image/attribute/worker triplet:

<image_id> <attribute_id> <is_present> <certainty_id> <time>

where <image_id>, <attribute_id>, <certainty_id> correspond to the IDs in images.txt, attributes/attributes.txt, and attributes/certainties.txt respectively.  <is_present> is 0 or 1 (1 denotes that the attribute is present).  <time> denotes the time spent by the MTurker in seconds.
-----------------------------------------------------------------------------------


------- Class attribute labels (attributes/class_attribute_labels_continuous.txt) ------
Attributes on a per-class level--in a similar format to the Animals With Attributes dataset--are contained in attributes/class_attribute_labels_continuous.txt.  The file contains 200 lines and 312 space-separated columns.  Each line corresponds to one class (in the same order as classes.txt) and each column contains one real-valued number corresponding to one attribute (in the same order as attributes.txt).  The number is the percentage of the time (between 0 and 100) that a human thinks that the attribute is present for a given class
----------------------------------------------------------------------------------------

'''

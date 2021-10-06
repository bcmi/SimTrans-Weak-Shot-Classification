import sys
import warnings

sys.path.append(".")
warnings.filterwarnings("ignore")

from data.helper import *
import os
import scipy.io as sio
import numpy as np

web2source = {'ATR-42': 'ATR-42', 'ATR-72': 'ATR-72', 'Airbus_A300B4': 'A300B4', 'Airbus_A310': 'A310',
              'Airbus_A318': 'A318', 'Airbus_A319': 'A319', 'Airbus_A320': 'A320', 'Airbus_A321': 'A321',
              'Airbus_A330-200': 'A330-200', 'Airbus_A330-300': 'A330-300', 'Airbus_A340-200': 'A340-200',
              'Airbus_A340-300': 'A340-300', 'Airbus_A340-500': 'A340-500', 'Airbus_A340-600': 'A340-600',
              'Airbus_A380': 'A380', 'Antonov_An-12': 'An-12', 'Beechcraft_1900': 'Beechcraft 1900',
              'Beechcraft_King_Air_B200': 'Model B200', 'Boeing_707-320': '707-320', 'Boeing_717': 'Boeing 717',
              'Boeing_727-200': '727-200', 'Boeing_737-200': '737-200', 'Boeing_737-300': '737-300',
              'Boeing_737-400': '737-400', 'Boeing_737-500': '737-500', 'Boeing_737-600': '737-600',
              'Boeing_737-700': '737-700', 'Boeing_737-800': '737-800', 'Boeing_737-900': '737-900',
              'Boeing_747-100': '747-100', 'Boeing_747-200': '747-200', 'Boeing_747-300': '747-300',
              'Boeing_747-400': '747-400', 'Boeing_757-200': '757-200', 'Boeing_757-300': '757-300',
              'Boeing_767-200': '767-200', 'Boeing_767-300': '767-300', 'Boeing_767-400': '767-400',
              'Boeing_777-200': '777-200', 'Boeing_777-300': '777-300', 'Bombardier_Global_Express': 'Global Express',
              'British_Aerospace_125': 'BAE-125', 'British_Aerospace_146-200': 'BAE 146-200',
              'British_Aerospace_146-300': 'BAE 146-300', 'British_Aerospace_Hawk_T1': 'Hawk T1',
              'Canadair_CRJ-200': 'CRJ-200', 'Canadair_CRJ-700': 'CRJ-700', 'Canadair_CRJ-900': 'CRJ-900',
              'Canadair_Challenger_600': 'Challenger 600', 'Cessna_172': 'Cessna 172', 'Cessna_208': 'Cessna 208',
              'Cessna_Citation_525': 'Cessna 525', 'Cessna_Citation_560': 'Cessna 560', 'Cirrus_SR-20': 'SR-20',
              'Dassault_Falcon_2000': 'Falcon 2000', 'Dassault_Falcon_900': 'Falcon 900', 'Dornier_328': 'Dornier 328',
              'Douglas_C-47': 'C-47', 'Douglas_DC-3': 'DC-3', 'Douglas_DC-6': 'DC-6', 'Douglas_DC-8': 'DC-8',
              'Embraer_E-170': 'E-170', 'Embraer_E-190': 'E-190', 'Embraer_E-195': 'E-195',
              'Embraer_EMB-120': 'EMB-120', 'Embraer_ERJ_135': 'ERJ 135', 'Embraer_ERJ_145': 'ERJ 145',
              'Embraer_Legacy_600': 'Embraer Legacy 600', 'Eurofighter_Typhoon': 'Eurofighter Typhoon',
              'Fairchild_Metroliner': 'Metroliner', 'Fokker_100': 'Fokker 100', 'Fokker_50': 'Fokker 50',
              'Fokker_70': 'Fokker 70', 'Gulfstream_IV': 'Gulfstream IV', 'Gulfstream_V': 'Gulfstream V',
              'Ilyushin_Il-76': 'Il-76', 'Lockheed_C-130': 'C-130', 'Lockheed_L-1011': 'L-1011',
              'Lockheed_Martin_F-16AB': 'F-16A/B', 'McDonnell_Douglas_DC-10': 'DC-10',
              'McDonnell_Douglas_DC-9-30': 'DC-9-30', 'McDonnell_Douglas_FA-18': 'F/A-18',
              'McDonnell_Douglas_MD-11': 'MD-11', 'McDonnell_Douglas_MD-80': 'MD-80',
              'McDonnell_Douglas_MD-87': 'MD-87', 'McDonnell_Douglas_MD-90': 'MD-90', 'Panavia_Tornado': 'Tornado',
              'Piper_PA-28': 'PA-28', 'Robin_DR-400': 'DR-400', 'Saab_2000': 'Saab 2000', 'Saab_340': 'Saab 340',
              'Supermarine_Spitfire': 'Spitfire', 'Tupolev_Tu-134': 'Tu-134', 'Tupolev_Tu-154': 'Tu-154',
              'Yakovlev_Yak-42': 'Yak-42', 'de_Havilland_DH-82': 'DH-82', 'de_Havilland_DHC-1': 'DHC-1',
              'de_Havilland_DHC-6': 'DHC-6', 'de_Havilland_DHC-8-100': 'DHC-8-100',
              'de_Havilland_DHC-8-300': 'DHC-8-300'}
source2web = {'ATR-42': 'ATR-42', 'ATR-72': 'ATR-72', 'A300B4': 'Airbus_A300B4', 'A310': 'Airbus_A310',
              'A318': 'Airbus_A318', 'A319': 'Airbus_A319', 'A320': 'Airbus_A320', 'A321': 'Airbus_A321',
              'A330-200': 'Airbus_A330-200', 'A330-300': 'Airbus_A330-300', 'A340-200': 'Airbus_A340-200',
              'A340-300': 'Airbus_A340-300', 'A340-500': 'Airbus_A340-500', 'A340-600': 'Airbus_A340-600',
              'A380': 'Airbus_A380', 'An-12': 'Antonov_An-12', 'Beechcraft 1900': 'Beechcraft_1900',
              'Model B200': 'Beechcraft_King_Air_B200', '707-320': 'Boeing_707-320', 'Boeing 717': 'Boeing_717',
              '727-200': 'Boeing_727-200', '737-200': 'Boeing_737-200', '737-300': 'Boeing_737-300',
              '737-400': 'Boeing_737-400', '737-500': 'Boeing_737-500', '737-600': 'Boeing_737-600',
              '737-700': 'Boeing_737-700', '737-800': 'Boeing_737-800', '737-900': 'Boeing_737-900',
              '747-100': 'Boeing_747-100', '747-200': 'Boeing_747-200', '747-300': 'Boeing_747-300',
              '747-400': 'Boeing_747-400', '757-200': 'Boeing_757-200', '757-300': 'Boeing_757-300',
              '767-200': 'Boeing_767-200', '767-300': 'Boeing_767-300', '767-400': 'Boeing_767-400',
              '777-200': 'Boeing_777-200', '777-300': 'Boeing_777-300', 'Global Express': 'Bombardier_Global_Express',
              'BAE-125': 'British_Aerospace_125', 'BAE 146-200': 'British_Aerospace_146-200',
              'BAE 146-300': 'British_Aerospace_146-300', 'Hawk T1': 'British_Aerospace_Hawk_T1',
              'CRJ-200': 'Canadair_CRJ-200', 'CRJ-700': 'Canadair_CRJ-700', 'CRJ-900': 'Canadair_CRJ-900',
              'Challenger 600': 'Canadair_Challenger_600', 'Cessna 172': 'Cessna_172', 'Cessna 208': 'Cessna_208',
              'Cessna 525': 'Cessna_Citation_525', 'Cessna 560': 'Cessna_Citation_560', 'SR-20': 'Cirrus_SR-20',
              'Falcon 2000': 'Dassault_Falcon_2000', 'Falcon 900': 'Dassault_Falcon_900', 'Dornier 328': 'Dornier_328',
              'C-47': 'Douglas_C-47', 'DC-3': 'Douglas_DC-3', 'DC-6': 'Douglas_DC-6', 'DC-8': 'Douglas_DC-8',
              'E-170': 'Embraer_E-170', 'E-190': 'Embraer_E-190', 'E-195': 'Embraer_E-195',
              'EMB-120': 'Embraer_EMB-120', 'ERJ 135': 'Embraer_ERJ_135', 'ERJ 145': 'Embraer_ERJ_145',
              'Embraer Legacy 600': 'Embraer_Legacy_600', 'Eurofighter Typhoon': 'Eurofighter_Typhoon',
              'Metroliner': 'Fairchild_Metroliner', 'Fokker 100': 'Fokker_100', 'Fokker 50': 'Fokker_50',
              'Fokker 70': 'Fokker_70', 'Gulfstream IV': 'Gulfstream_IV', 'Gulfstream V': 'Gulfstream_V',
              'Il-76': 'Ilyushin_Il-76', 'C-130': 'Lockheed_C-130', 'L-1011': 'Lockheed_L-1011',
              'F-16A/B': 'Lockheed_Martin_F-16AB', 'DC-10': 'McDonnell_Douglas_DC-10',
              'DC-9-30': 'McDonnell_Douglas_DC-9-30', 'F/A-18': 'McDonnell_Douglas_FA-18',
              'MD-11': 'McDonnell_Douglas_MD-11', 'MD-80': 'McDonnell_Douglas_MD-80',
              'MD-87': 'McDonnell_Douglas_MD-87', 'MD-90': 'McDonnell_Douglas_MD-90', 'Tornado': 'Panavia_Tornado',
              'PA-28': 'Piper_PA-28', 'DR-400': 'Robin_DR-400', 'Saab 2000': 'Saab_2000', 'Saab 340': 'Saab_340',
              'Spitfire': 'Supermarine_Spitfire', 'Tu-134': 'Tupolev_Tu-134', 'Tu-154': 'Tupolev_Tu-154',
              'Yak-42': 'Yakovlev_Yak-42', 'DH-82': 'de_Havilland_DH-82', 'DHC-1': 'de_Havilland_DHC-1',
              'DHC-6': 'de_Havilland_DHC-6', 'DHC-8-100': 'de_Havilland_DHC-8-100',
              'DHC-8-300': 'de_Havilland_DHC-8-300'}


def read_split_from_file(fpath):
    classes = []
    for line in open(fpath).readlines():
        classes.append(line.strip())
    return classes


class AirHelper(DataHelper):
    def __init__(self, args):
        super(AirHelper, self).__init__(args)
        all_categories = read_split_from_file(self.root_path + '/fgvc-aircraft-2013b/data/variants.txt')

        names = ['727-200', '737-200', '757-300', '767-200', 'A310', 'A321', 'A340-600', 'ATR-72', 'An-12',
                 'Beechcraft 1900',
                 'C-130', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525', 'DHC-1', 'DHC-6', 'DR-400',
                 'Dornier 328', 'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A/B', 'F/A-18', 'Falcon 900',
                 'Fokker 50',
                 'Fokker 70', 'Global Express', 'Gulfstream IV', 'Gulfstream V', 'Model B200', 'PA-28', 'SR-20',
                 'Saab 2000',
                 'Saab 340', 'Spitfire', 'Tornado', 'Tu-154']
        # np.random.permutation(len(names))[:25].tolist()
        self.novel_categories = [names[c] for c in [22, 20, 16, 10, 31, 28, 15, 11, 2, 25, 30, 32, 36, 34, 29, 33, 8, 13, 5, 17, 14, 7, 27, 1, 12]]

        self.base_categories = []
        for c in all_categories:
            if c not in self.novel_categories:
                self.base_categories.append(c)

        self.name = 'Air'

    def _get_novel_test_set(self):
        return SourceTestSet(self.root_path, self.novel_categories, self.test_transforms)

    def _get_base_test_set(self):
        return SourceTestSet(self.root_path, self.base_categories, self.test_transforms)

    def _get_clean_novel_set(self):
        return SourceTrainSet(self.root_path, self.novel_categories, self.train_transforms)

    def _get_clean_base_set(self):
        return SourceTrainSet(self.root_path, self.base_categories, self.train_transforms)

    def _get_noisy_novel_set(self):
        return NoisySet(self.root_path, self.novel_categories, self.train_transforms)

    def _get_noisy_base_set(self):
        raise NotImplementedError
        return NoisySet(self.root_path, self.novel_web_names, self.train_transforms)


class SourceTrainSet(DataSet):
    def __init__(self, root_path, categories, transform=None):
        super(SourceTrainSet, self).__init__(root_path, categories, transform)
        fpath = self.root_path + '/fgvc-aircraft-2013b/data/images_variant_trainval.txt'

        self.image_list = []
        for line in open(fpath).readlines():
            row = line.strip()
            img_id = row[:7]
            image_path = self.root_path + '/fgvc-aircraft-2013b/data/images/' + img_id + '.jpg'
            assert os.path.exists(image_path), f'image not found: {image_path}'
            category = row[8:]

            if category in self.categories:
                self.image_list.append((image_path, self.category2int[category]))

        return


class SourceTestSet(DataSet):
    def __init__(self, root_path, categories, transform=None):
        super(SourceTestSet, self).__init__(root_path, categories, transform)

        fpath = self.root_path + '/fgvc-aircraft-2013b/data/images_variant_test.txt'

        self.image_list = []
        for line in open(fpath).readlines():
            row = line.strip()
            img_id = row[:7]
            image_path = self.root_path + '/fgvc-aircraft-2013b/data/images/' + img_id + '.jpg'
            assert os.path.exists(image_path), f'image not found: {image_path}'
            category = row[8:]

            if category in self.categories:
                self.image_list.append((image_path, self.category2int[category]))

        return


class NoisySet(DataSet):
    def __init__(self, root_path, categories, transform=None, max_noisy_images_per=None):
        super(NoisySet, self).__init__(root_path, categories, transform)

        self.image_list = []
        for category_name in categories:
            dir_path = f'{self.root_path}/hardest_new_set/{category_name}'
            assert os.path.exists(dir_path), f'not found: {dir_path}'

            image_names = sorted(os.listdir(dir_path))
            category_list = [(os.path.join(dir_path, image_name),
                              self.category2int[category_name]) for
                             image_name in image_names]

            self.image_list += category_list[:max_noisy_images_per]

        lengths = [len(v) for v in self.get_dict().values()]
        print(lengths)
        return


# class NoisySet(DataSet):
#     def __init__(self, root_path, categories, transform=None, max_noisy_images_per=None):
#         super(NoisySet, self).__init__(root_path, categories, transform)
#
#         self.image_list = []
#         for category_name in categories:
#             web_name = source2web[category_name]
#
#             dir_path1 = f'{self.root_path}/img_week1/{web_name}/downloads/img/{web_name}'
#             dir_path2 = f'{self.root_path}/img_zip2/{web_name}/downloads/img/{web_name}'
#
#             category_list = []
#             if os.path.exists(dir_path1):
#                 category_list += [(os.path.join(dir_path1, image_name),
#                                    self.category2int[category_name]) for
#                                   image_name in sorted(os.listdir(dir_path1))]
#
#             if os.path.exists(dir_path2):
#                 category_list += [(os.path.join(dir_path2, image_name),
#                                    self.category2int[category_name]) for
#                                   image_name in sorted(os.listdir(dir_path2))]
#
#             self.image_list += category_list
#
#         # lengths = {k: len(v) for k, v in self.get_dict().items()}
#         #
#         # needs = {}
#         # for k, v in lengths.items():
#         #     if v < 1000:
#         #         needs[source2web[k]] = v
#
#         return


if __name__ == '__main__':
    import os
    from tqdm import tqdm

    root_path = '../dataset/Air'

    all_categories = selected_web_name = sorted(os.listdir(f'{root_path}/img_zip2'))

    dataset = NoisySet(root_path, all_categories)
    for p, c in tqdm(dataset.image_list):
        try:
            image = Image.open(p).convert('RGB')
        except:
            print(f'remove: {p}')
            os.remove(p)
    d = 1

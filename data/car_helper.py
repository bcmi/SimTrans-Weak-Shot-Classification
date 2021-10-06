from data.helper import *
import os
import scipy.io as sio


class CarHelper(DataHelper):
    def __init__(self, args):
        super(CarHelper, self).__init__(args)
        self.name = 'Car'
        all_categories = sorted(os.listdir(os.path.join(self.root_path, 'web_data')))

        if args.novel_split < 0:
            print('default split')
            self.base_categories = []
            self.novel_categories = []
            for i, c in enumerate(all_categories):
                if i % 4 == 0:
                    self.novel_categories.append(c)
                else:
                    self.base_categories.append(c)
        else:
            all_c = ['Arrizo 7', 'Yuexiang hatchback', 'Volvo S60', 'Passat Lingyu', 'Sharan', 'Zhonghua H230',
                     'BAW E Series hatchback', 'Zhonghua Zunchi', 'BYD F0', 'ASX abroad version', 'Audi Q5', 'Lexus CT',
                     'SAAB D70', 'MG6 hatchback', 'Lexus IS', 'BWM 2 Series', 'Peugeot 207 sedan', 'Nissan NV200',
                     'Lexus GS', 'Mazda 3 abroad version', 'Impreza hatchback', 'Gallardo', 'Lexus RX', 'smart fortwo',
                     'Chrey QQ', 'Benz SLK Class', 'Mazda CX-5', 'BWM M5', 'Benz S Class', 'Baojun 610', 'Chuanqi GS5',
                     'Honda CR-V', 'Peugeot 308SW', 'Lusheng E70', 'Classic Imperial hatchback', 'Golf GTI', 'Zafira',
                     'BWM 3 Series GT', 'BWM X4', 'Golf convertible', 'Passat', 'Mitsubishi Lancer EX', 'Besturn X80',
                     'Alphard', 'Landwind X8', 'Volvo XC90 abroad version', 'Geely EC8', 'Tiguan abroad version',
                     'Tongyue', 'Audi TTS coupe', 'Ruifeng S5', 'Livina', 'Mitsubishi Fortis', 'Volvo XC60',
                     'Audi A5 coupe', 'BYD G3', 'Peugeot 508', 'Volvo S40', 'MG3', 'Brabus S Class', 'Cadillac SRX',
                     'Lechi', 'Cowin 2', 'Haorui', 'Eastar Cross', 'Benben MINI', 'Audi A8L', 'Sportage',
                     'Cruze hatchback', 'Venza', 'Polo sedan', 'Sorento', 'Golf estate', 'Audi S8', 'Touran', 'Moinca',
                     'Audi A5 convertible', 'Soul', 'Zhonghua V5', 'Golf', 'Audi A3 sedan', 'Yeti', 'Heyue RS',
                     'Exclle', 'Chrey A3 sedan', 'Infiniti QX50', 'Mazda 5', 'Crown', 'Evoque', 'Youyou', 'Yuexiang V5',
                     'EZ', 'New Focus hatchback', 'EXCELLE  XT', 'Chrey X1', 'Feixiang', 'BWM 5 Series GT', 'Lifan 720',
                     'MINI COUNTRYMAN', 'Cowin 3', 'Nissan GT-R', 'Superb', 'Haima S7', 'Forte', 'KIA K2 sedan',
                     'New Focus sedan', 'Xiali N5', 'Sylphy', 'King Kong sedan', 'Zhixiang', 'Panamera',
                     'Great Wall M2', 'Tiida', 'Yuyan', 'Peugeot 207CC', 'Antara', 'Baojun 630', 'Lexus GS hybrid',
                     'BWM 4 Series convertible', 'Lifan 520', 'MG3 SW', 'Ruifeng M5', 'BWM 3 Series', 'Audi A4 estate',
                     'Outlander abroad version', 'Impreza sedan', 'Volkswagen Eos', 'Zhonghua Junjie FSV',
                     'Benz R Class', 'Peugeot 307 hatchback', 'Reiz', 'Gaguar XF', 'Magotan estate', 'Qiubite',
                     'Infiniti Q70L', 'Audi Q3', 'Cultus', 'Roewe 550', 'BYD L3', 'Benz C Class estate', 'Tiggo 5',
                     'Elysee', 'Range Rover Sport', 'Classic Imperial sedan', 'Audi A3 hatchback', 'Volvo C30',
                     'Fulwin 2 hatchback', 'Classic Focus sedan', 'Besturn B90', 'Grandtiger TUV', 'Peugeot RCZ',
                     'Bora', 'Gaguar XJ', 'Fengshen CROSS', 'Kazishi', 'Benz E Class couple', 'BWM 1 Series hatchback',
                     'Benz C Class AMG', 'Bravo', 'Mazda 3 Xingcheng sedan', 'Camaro', 'Zhonghua H530', 'Crosstour',
                     'Lexus GX', 'Volvo S80L', 'BWM 3 Series estate', 'Infiniti G Class', 'Audi A7', 'Kyron',
                     'BWM 6 Series', 'BYD F6', 'Sail hatchback', 'Qoros 3', 'Heyue', 'Sagitar', 'Zhonghua Junjie',
                     'Tiguan', 'Mazda 3', 'Beetle', 'Focus ST', 'Zhonghua Junjie FRV', 'Fulwin 2 sedan',
                     'BWM 3 Series coupe', 'Ruiyi coupe', 'Enclave', 'Mazda 6', 'BWM X3', 'Qijian A9', 'Quatre sedan',
                     'c-Elysee sedan', 'Canyenne', 'DS 4', 'Mustang', 'Cruze sedan', 'Jingyue', 'Tianyushangyue',
                     'Haval H3', 'Premacy', 'Citroen C5', 'Ruiyi', 'Regal GS', 'Roewe 750', 'Peugeot 408',
                     'Audi A5 hatchback', 'Lavide', 'Epica', 'MINI PACEMAN', 'Volvo V60', 'Fiesta sedan',
                     'Elantra Yuedong', 'Prado', 'BWM 5 Series', 'Great Wall C50', 'Haima S5', 'Alto', 'KIA K5',
                     'Peugeot 207 hatchback', 'Fengshen H30', 'Trax', 'Geely SC7', 'Shuaike', 'Weizhi', 'Toyota RAV4',
                     'Zhiyue', 'Cadillac ATS-L', 'Sunshine', 'Zhishang XT', 'Ruiying', 'Yuexiang sedan', 'Wrangler',
                     'Changan CS35', 'Wingle 5', 'Lexus ES hybrid', 'Mazda 3 Xingcheng hatchback', 'Audi A4L',
                     'Touareg', 'Ruiqi G5', 'BAW E Series sedan', 'BWM X6', 'Peugeot 4008', 'Kaizun', 'Mazda 2',
                     'Dahaishi', 'MAXUS V80xs', 'DS 3', 'Family M5', 'Ruiping', 'BWM 3 Series convertible',
                     'Cadillac XTS', 'Tiggo', 'Cross Polo', 'FIAT 500', 'Besturn B70', 'Yongyuan A380', 'Spirior',
                     'Wulingrongguang', 'Huaguan', 'Mistra', 'Cadillac CTS', 'Citroen C4L', 'Discovery', 'Magotan',
                     'Avante', 'Great Wall V80', 'Atenza', 'Sportage R', 'Fengshen A60', 'Peugeot 307 sedan', 'Camry',
                     'MG7', 'Infiniti QX70', 'Polo hatchback', 'Tianyu SX4 hatchback', 'Porsche 911', 'Teana',
                     'Santana', 'Cayman', 'Grandis', 'Sail sedan', 'New Sylphy', 'Lexus RX hybrid', 'Great Wall M4',
                     'Lifan 320', 'Benz GLK Class', 'Benz GL Class', 'MG6 sedan', 'Zhonghua Junjie Wagon', 'Roewe 350',
                     'Fiesta hatchback', 'Infiniti Q50', 'Mazda CX7', 'BYD F3R', 'Range Rover',
                     'Buick GL8 Luxury Business', 'Peugeot 308', 'Ziyoujian', 'Yuexiang V3', 'BYD S6', 'Cross Lavida',
                     'Zotye 5008', 'Zhonghua H330', 'Multivan', 'Rapid', 'Buick GL8 Business', 'Weizhi V5',
                     'EXCELLE  GT', 'KIA K3', 'Aveo sedan', 'Jetta', 'Axela sedan', 'Binyue', 'Haimaqishi', 'Equus',
                     'Vios', 'Infiniti QX80', 'Xuanli', 'Rena', 'Ma Chi', 'Gran Lavida', 'Saboo GX5', 'Mendeo Zhisheng',
                     'Ecosport', 'Chrey E5', 'Scenic', 'Audi S5 convertible', 'Benben LOVE', 'BWM X1', 'Octavia',
                     'Sonata 8', 'Benben', 'Malibu', 'Quatre hatchback', 'Rohens', 'Haydo', 'Qiteng M70', 'Wind Lang',
                     'Peugeot 301', 'S-MAX', 'encore', 'Linian S1', 'Aveo', 'DS 5LS', 'Chrysler 300C', 'Peugeot 308CC',
                     'Jade', 'Civic', 'Audi Q7', 'Lacrosse', 'WeaLink X5', 'i30', 'MINI', 'KIA K3S',
                     'Volvo V40 CrossCountry', 'Accord', 'Peugeot 2008', 'Sonata', 'Veloster', 'Benz A Class',
                     'Landwind X5', 'Audi A1', 'Lova', 'Classic Focus hatchback', 'Scirocco', 'Zhonghua Junjie CROSS',
                     'ix35', 'Yaris', 'Pajero', 'BWM 7 Series', 'Captiva', 'Haima M3', 'Citroen C2', 'Wulingzhiguang',
                     'Chrey A3 hatchback', 'Fengshen S30', 'Jingyi', 'DS 5', 'AVEO hatchback', 'Benz G Class AMG',
                     'Volvo S60L', 'Benz E Class convertible', 'Changan CX20', 'BWM X5', 'Volkswagen CC', 'Venucia R50',
                     'Mazda 2 sedan', 'Latitude', 'Benz E Class', 'Benz C Class', 'MINI CLUBMAN', 'Peugeot 3008',
                     'Venucia D50', 'Lexus LS hybrid', 'Lingzhi', 'Volvo C70', 'Weizhi V2', 'Superb Derivative',
                     'Besturn B50', 'Lcruiser', 'Seville SLS', 'Prius', 'Tianyu SX4 sedan', 'BYD M6', 'Crider',
                     'Borrego', 'Grandtiger G3', 'Evo', 'Patriot', 'Shuma', 'Camry hybrid', 'Lexus IS convertible',
                     'Qashqai', 'Koleos', 'Lexus ES', 'Audi A6L', 'Compass', 'Variant', 'Volvo V40', 'Chrey QQ3',
                     'Verna', 'Phaeton', 'Toyota 86', 'GTC hatchback', 'Audi TT coupe', 'Bei Douxing', 'Yidong',
                     'Regal', 'Santafe', 'Fabia', 'Wulinghongguang', 'Xuanli CROSS', 'MG5']
            self.base_categories = all_c[args.novel_split:]
            self.novel_categories = all_c[:args.novel_split]

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
        return NoisySet(self.root_path, self.novel_categories, self.train_transforms)

    def _get_noisy_base_set(self):
        return NoisySet(self.root_path, self.base_categories, self.train_transforms)


def read_from_file(fpath):
    classes = []
    for line in open(fpath).readlines():
        classes.append(line.strip())
    return classes


class SourceTrainSet(DataSet):
    def __init__(self, root_path, categories, transform=None):
        super(SourceTrainSet, self).__init__(root_path, categories, transform)
        info = sio.loadmat(os.path.join(root_path, 'CampCar/misc/make_model_name.mat'))

        id_to_name = {}
        for i, name in enumerate(info['model_names']):
            idx = i + 1
            if len(name[0]) > 0:
                id_to_name[idx] = name[0][0].strip()
            else:
                id_to_name[idx] = 'unknown'
        all_train_images = read_from_file(os.path.join(root_path, 'CampCar/train_test_split/classification/train.txt'))

        self.image_list = []

        for ipath in all_train_images:
            name = id_to_name[int(ipath.split('/')[1])]
            if name in categories:
                self.image_list.append((os.path.join(root_path, 'CampCar/image', ipath), self.category2int[name]))

        # data_dict = self.get_dict()
        # data_keys = list(data_dict.keys())
        # for c in categories:
        #     if c not in data_keys:
        #         d = 1
        return


class SourceTestSet(DataSet):
    def __init__(self, root_path, categories, transform=None):
        super(SourceTestSet, self).__init__(root_path, categories, transform)
        info = sio.loadmat(os.path.join(root_path, 'CampCar/misc/make_model_name.mat'))

        id_to_name = {}
        for i, name in enumerate(info['model_names']):
            idx = i + 1
            if len(name[0]) > 0:
                id_to_name[idx] = str(name[0][0]).lstrip().rstrip()
            else:
                id_to_name[idx] = 'unknown'

        name_to_id = {}
        for idx, name in id_to_name.items():
            if name != 'unknown' and name in categories:
                name_to_id[name] = idx

        for c in categories:
            if c not in id_to_name.values():
                print(c)

        all_test_images = read_from_file(os.path.join(root_path, 'CampCar/train_test_split/classification/test.txt'))

        self.image_list = []

        for ipath in all_test_images:
            name = id_to_name[int(ipath.split('/')[1])]
            if name in categories:
                self.image_list.append((os.path.join(root_path, 'CampCar/image', ipath), self.category2int[name]))
        return


class NoisySet(DataSet):
    def __init__(self, root_path, categories, transform=None, max_noisy_images_per=None):
        super(NoisySet, self).__init__(root_path, categories, transform)

        self.image_list = []
        for category_name in categories:
            dir_path = os.path.join(self.root_path, 'web_data', category_name)
            if not os.path.exists(dir_path):
                print(category_name)
                print(dir_path)
                raise FileNotFoundError

            image_names = sorted(os.listdir(dir_path))
            category_list = [(os.path.join(dir_path, image_name),
                              self.category2int[category_name]) for
                             image_name in image_names]

            self.image_list += category_list[:max_noisy_images_per]

        return

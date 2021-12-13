from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from RandomErasing import RandomErasing
from RandomSampler import RandomSampler
import os
import re

BATCH_FOR_ID = 5
BATCH_PER_ID = 4
T_BATCH_SIZE = 128
Q_BATCH_SIZE = 128
G_BATCH_SIZE = 128


class Data():
    def __init__(self, parent_dir):
        train_transform = transforms.Compose([
            transforms.Resize((128, 384)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((128, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = Market1501(train_transform, 'train', parent_dir)
        self.testset = Market1501(test_transform, 'test', parent_dir)
        self.queryset = Market1501(test_transform, 'query', parent_dir)

        self.train_loader = dataloader.DataLoader(self.trainset,
                                                  sampler=RandomSampler(self.trainset, batch_id=BATCH_FOR_ID,
                                                                        batch_image=BATCH_PER_ID),
                                                  batch_size=BATCH_PER_ID * BATCH_FOR_ID, num_workers=8,
                                                  pin_memory=True)

        self.test_loader = dataloader.DataLoader(self.testset, batch_size=G_BATCH_SIZE, num_workers=8,
                                                 pin_memory=True)
        self.query_loader = dataloader.DataLoader(self.queryset, batch_size=Q_BATCH_SIZE, num_workers=8,
                                                  pin_memory=True)


class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):
        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path += '/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(os.path.basename(file_path).split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(os.path.basename(file_path).split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):

        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])

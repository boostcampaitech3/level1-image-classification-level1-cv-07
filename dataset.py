import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List, Sequence, Optional, Union

import cv2
import numpy as np
import torch
import colors
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            CenterCrop((320, 256)),
            Resize(resize, Image.BILINEAR),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        # self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestDataset(Dataset):
    def __init__(self, img_paths, transform, use_PIL):
        self.img_paths = img_paths
        self.transform = transform
        self.use_PIL = use_PIL

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        if self.use_PIL:
            image = Image.open(img_path)
        else:
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


# TODO: train / valid dataset
# 0. 이미지 단위 대신 인물 단위로 데이터셋 분할
#   이미지 단위로 분할: gender, age 추론에 대해서는 validation 데이터셋이 오염된 것으로 취급할 수 있다고 생
# 1. 출력 선택
#   class 0~17 / [mask: 0~2, gender: 0~1, age: 0~2] / 개별 속성 (mask: 0~2 | gender: 0~1 | age: 0~2)
# 2. 입력 선택
#   인물별 디렉토리 경로 모음 / csv / ...


class CustomDatasetSplitByProfile(MaskBaseDataset):
    """마스크 데이터셋

    주어진 텍스트 파일에서 프로필을 읽어 데이터셋 구성.
    output 파라미터를 통해 데이터셋 라벨 출력 설정.
    데이터 로드시에 이미지의 RGB mean, std를 계산하여 출력. 복사하여 재사용 추천.

    Args:
        data_dir:
            profile 디렉토리를 포함하는 디렉토리 경로
        profiles_file:
            각 줄에 profile(ex - 006477_female_Asian_18)을 포함하는 텍스트파일 경로
        output:
            데이터셋 출력
                * class: 0~17
                * mask | gender | age: 각 속성의 label, 0~2 or 0~1
                * all: (mask, gender, age)
        mean:
            이미지 RGB mean / None인 경우 계산
        std:
            이미지 RGB std / None인 경우 계산
        use_PIL:
            이미지를 PIL로 읽음. False인 경우 opencv로 읽음.
        pass_calc_statistics:
            True인 경우 이미지 RGB mean, std 계산을 pass / validation 셋에서 생략하기 위함.
    """
    def __init__(self,
                 data_dir: str,
                 profiles_file: str,
                 output: str = 'all',
                 mean: Optional[Sequence] = None,
                 std: Optional[Sequence] = None,
                 pass_calc_statistics: bool = False,
                 use_PIL: bool = True,
                 ):
        if output not in ('all', 'mask', 'gender', 'age', 'class'):
            raise ValueError(f'정의되지 않은 output: {output} / (all|mask|gender|age|class)')
        self.output = output
        self.profiles_file = profiles_file
        self.aug_by_torchvision = use_PIL
        super().__init__(data_dir, mean, std, None)

        if not pass_calc_statistics:
            self.calc_statistics()

    def calc_statistics(self):
        super().calc_statistics()
        print(colors.red('mean:'), tuple(self.mean))
        print(colors.red('std:'), tuple(self.std))

    def read_image(self, index):
        if self.aug_by_torchvision:
            return super().read_image(index)
        else:
            image_path = self.image_paths[index]
            return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    def setup(self):
        """profile 파일로부터 이미지 경로, 마스크, 성별, 연령대 라벨 획득"""
        self.image_paths = []
        self. mask_labels = []
        self.gender_labels = []
        self.age_labels = []

        with open(self.profiles_file, 'r') as f:  # 텍스트 파일 로드
            profiles = [line.strip('\n') for line in f.readlines()]

        for profile in profiles:
            _, gender, _, age = profile.split('_')
            gender_label = GenderLabels.from_str(gender)
            age_label = AgeLabels.from_number(age)

            img_dir = os.path.join(self.data_dir, profile)
            for img_name in os.listdir(img_dir):
                stem, ext = os.path.splitext(img_name)
                if stem not in self._file_names:  # 정의된 파일명 외에는 제거
                    continue

                image_path = os.path.join(self.data_dir, profile, img_name)
                mask_label = self._file_names[stem]

                self.image_paths.append(image_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)

        if self.output == 'class':
            label = self.encode_multi_class((mask_label, gender_label, age_label))
        elif self.output == 'all':
            label = torch.tensor([mask_label, gender_label, age_label])
        elif self.output == 'mask':
            label = mask_label
        elif self.output == 'gender':
            label = gender_label
        else:
            label = age_label

        image_transform = self.transform(image)
        return image_transform, label

    @staticmethod
    def encode_multi_class(decoded: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
        with torch.no_grad():
            if not isinstance(decoded, torch.Tensor):
                decoded = torch.tensor(decoded)

            if decoded.shape[-1] == 3:  # label
                mask_label, gender_label, age_label = map(torch.squeeze, torch.split(decoded, [1, 1, 1], -1))
            elif decoded.shape[-1] == 8:  # prediction
                mask_pred, gender_pred, age_pred = torch.split(decoded, [3, 2, 3], -1)
                mask_label = mask_pred.argmax(-1).squeeze()
                gender_label = gender_pred.argmax(-1).squeeze()
                age_label = age_pred.argmax(-1).squeeze()
            else:
                raise ValueError(f'정의되지 않은 입력 차원: {decoded.shape[1]} / (3|8)')

        return mask_label * 6 + gender_label * 3 + age_label


class ATransform:
    """albumentation 이용시에 상속하여 transform 정의"""
    def __call__(self, x):
        return self.transform(image=x)['image']


class Aug0(ATransform):
    """albumentation 예시"""
    def __init__(self, resize, mean, std, is_train=True):
        if is_train:
            self.transform = A.Compose([
                A.Resize(*resize),
                A.HorizontalFlip(p=.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.75),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.75),
                A.RandomBrightnessContrast(p=0.75),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*resize),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])


if __name__ == '__main__':
    import numpy as np

    DATA_DIR, PROFILES_FILE = '/opt/ml/input/data/train/images', 'profile_test.txt'

    dataset_output_all = CustomDatasetSplitByProfile(
        data_dir=DATA_DIR,
        profiles_file=PROFILES_FILE,
        output='all',
    )
    dataset_output_all.set_transform(lambda x: np.array(x).shape)
    print('output all:', next(iter(dataset_output_all)), '\n')

    dataset_output_mask = CustomDatasetSplitByProfile(
        data_dir=DATA_DIR,
        profiles_file=PROFILES_FILE,
        output='mask',
    )
    dataset_output_mask.set_transform(lambda x: np.array(x).shape)
    print('output mask:', next(iter(dataset_output_mask)), '\n')

    dataset_output_class = CustomDatasetSplitByProfile(
        data_dir=DATA_DIR,
        profiles_file=PROFILES_FILE,
        output='class',
    )
    dataset_output_class.set_transform(lambda x: np.array(x).shape)
    print('output class:', dataset_output_class[0], '\n')





























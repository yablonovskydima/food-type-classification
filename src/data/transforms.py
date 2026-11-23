from collections.abc import Sequence

from torchvision import transforms

from src.data.constants import DEFAULT_IMAGE_SIZE, FOOD11_MEAN, FOOD11_STD


def get_train_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = FOOD11_MEAN,
    std: Sequence[float] = FOOD11_STD,
    randaugment: bool = True,
    random_erasing_p: float = 0.2,
    randaugment_magnitude: int = 9,
    crop_min_scale: float = 0.7,
) -> transforms.Compose:
    transform_list: list[object] = [
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(crop_min_scale, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ]

    if randaugment:
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=randaugment_magnitude))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if random_erasing_p > 0:
        transform_list.append(
            transforms.RandomErasing(
                p=random_erasing_p,
                scale=(0.02, 0.2),
                value="random",
            )
        )

    return transforms.Compose(transform_list)


def get_eval_geometry_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    crop_scale: float = 256.0 / 224.0,
) -> transforms.Compose:
    resize_dim = int(round(image_size * crop_scale))
    return transforms.Compose([
        transforms.Resize(
            resize_dim,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(image_size),
    ])


def get_eval_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: Sequence[float] = FOOD11_MEAN,
    std: Sequence[float] = FOOD11_STD,
    crop_scale: float = 256.0 / 224.0,
) -> transforms.Compose:
    geometry = get_eval_geometry_transforms(image_size=image_size, crop_scale=crop_scale)
    return transforms.Compose([
        *geometry.transforms,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

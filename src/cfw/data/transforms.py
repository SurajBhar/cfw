"""Data augmentation transforms for CFW training.

Responsibilities:
- Define dataset, transform, sampler, and dataloader building blocks.
- Support baseline and CFW data-loading behaviors across splits.
- Provide reusable data APIs for local and distributed training.
"""


from typing import Optional, Sequence, Tuple

import torch
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to PIL images with specified probability.

    Args:
        p: Probability of applying the blur (default: 0.5)
        radius_min: Minimum blur radius/sigma (default: 0.1)
        radius_max: Maximum blur radius/sigma (default: 2.0)
        kernel_size: Kernel size for Gaussian blur (default: 9)

    Example:
        >>> blur_transform = GaussianBlur(p=0.5, radius_min=0.1, radius_max=2.0)
        >>> blurred_image = blur_transform(image)
    """

    def __init__(
        self,
        *,
        p: float = 0.5,
        radius_min: float = 0.1,
        radius_max: float = 2.0,
        kernel_size: int = 9,
        sigma: Optional[Tuple[float, float]] = None,
    ):
        """Initialize blur augmentation with probability and sigma range."""
        # Backward-compatible alias used by tests and legacy callers.
        if sigma is not None:
            radius_min, radius_max = sigma

        self.kernel_size = kernel_size
        self.sigma = (radius_min, radius_max)

        transform = transforms.GaussianBlur(
            kernel_size=kernel_size,
            sigma=(radius_min, radius_max)
        )
        super().__init__(transforms=[transform], p=p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert PIL Image or numpy array to tensor, or keep as-is if already a tensor.

    This is useful when composing transforms where some inputs may already be tensors.

    Returns:
        Tensor: Converted image or original tensor

    Example:
        >>> to_tensor = MaybeToTensor()
        >>> # Works with PIL images
        >>> tensor1 = to_tensor(pil_image)
        >>> # Also works if input is already a tensor
        >>> tensor2 = to_tensor(existing_tensor)  # Returns existing_tensor unchanged
    """

    def __call__(self, pic):
        """
        Convert image to tensor if not already.

        Args:
            pic: PIL Image, numpy.ndarray, or torch.Tensor

        Returns:
            Tensor: Converted or original tensor
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# ImageNet normalization constants (standard for pre-trained models)
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    """
    Create normalization transform.

    Args:
        mean: Per-channel mean for normalization (default: ImageNet mean)
        std: Per-channel std for normalization (default: ImageNet std)

    Returns:
        Normalize transform

    Example:
        >>> normalize = make_normalize_transform()
        >>> normalized_image = normalize(image_tensor)
    """
    return transforms.Normalize(mean=mean, std=std)


def make_classification_train_transform(
    *,
    resize_size: Optional[int] = None,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    vflip_prob: float = 0.0,
    horizontal_flip: Optional[bool] = None,
    vertical_flip: Optional[bool] = None,
    color_jitter: bool = False,
    jitter_brightness: float = 0.4,
    jitter_contrast: float = 0.4,
    jitter_saturation: float = 0.4,
    jitter_hue: float = 0.1,
    gaussian_blur: bool = False,
    blur_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    """
    Create training transforms with data augmentation.

    This roughly matches torchvision's preset for classification training
    with additional configurable augmentations.

    Args:
        crop_size: Size of random crop (default: 224)
        interpolation: Interpolation mode (default: BICUBIC)
        hflip_prob: Probability of horizontal flip (default: 0.5)
        vflip_prob: Probability of vertical flip (default: 0.0)
        color_jitter: Whether to apply color jittering (default: False)
        gaussian_blur: Whether to apply Gaussian blur (default: False)
        blur_prob: Probability of applying blur if enabled (default: 0.5)
        mean: Normalization mean (default: ImageNet mean)
        std: Normalization std (default: ImageNet std)

    Returns:
        Composed transform pipeline for training

    Example:
        >>> train_transform = make_classification_train_transform(
        ...     crop_size=224,
        ...     hflip_prob=0.5,
        ...     color_jitter=True,
        ...     gaussian_blur=True
        ... )
    """
    transforms_list = []

    if resize_size is not None:
        transforms_list.append(
            transforms.Resize(resize_size, interpolation=interpolation)
        )

    transforms_list.append(
        transforms.RandomResizedCrop(crop_size, interpolation=interpolation)
    )

    # Backward-compatible bool flags used by legacy tests/callers.
    if horizontal_flip is not None:
        hflip_prob = 0.5 if horizontal_flip else 0.0
    if vertical_flip is not None:
        vflip_prob = 0.5 if vertical_flip else 0.0

    # Horizontal flip
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))

    # Vertical flip (useful for some datasets)
    if vflip_prob > 0.0:
        transforms_list.append(transforms.RandomVerticalFlip(vflip_prob))

    # Color jittering
    if color_jitter:
        transforms_list.append(
            transforms.ColorJitter(
                brightness=jitter_brightness,
                contrast=jitter_contrast,
                saturation=jitter_saturation,
                hue=jitter_hue,
            )
        )

    # Gaussian blur
    if gaussian_blur:
        transforms_list.append(GaussianBlur(p=blur_prob))

    # Final transforms: to tensor and normalize
    transforms_list.extend([
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ])

    return transforms.Compose(transforms_list)


def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    """
    Create evaluation transforms without data augmentation.

    This matches torchvision's preset for classification evaluation.
    No random operations, just resize, center crop, and normalize.

    Args:
        resize_size: Size to resize shorter edge (default: 256)
        interpolation: Interpolation mode (default: BICUBIC)
        crop_size: Size of center crop (default: 224)
        mean: Normalization mean (default: ImageNet mean)
        std: Normalization std (default: ImageNet std)

    Returns:
        Composed transform pipeline for evaluation

    Example:
        >>> eval_transform = make_classification_eval_transform(
        ...     resize_size=256,
        ...     crop_size=224
        ... )
    """
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


def make_no_aug_transform(
    *,
    resize_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    """
    Create minimal transforms without augmentation (resize + normalize only).

    Useful for feature extraction or when augmentation is not desired.

    Args:
        resize_size: Size to resize image (default: 224)
        mean: Normalization mean (default: ImageNet mean)
        std: Normalization std (default: ImageNet std)

    Returns:
        Composed transform pipeline with minimal processing

    Example:
        >>> no_aug_transform = make_no_aug_transform(resize_size=224)
    """
    transforms_list = [
        transforms.Resize((resize_size, resize_size)),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

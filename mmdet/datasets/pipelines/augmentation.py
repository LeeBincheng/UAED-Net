import numpy as np

from ..builder import PIPELINES
import mmcv
import os.path as osp


@PIPELINES.register_module()
class LoadAugmentedImageFromFile:
    """加载增强图像的Pipeline类."""

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results.get('aug_img_prefix', None) is not None:
            # 加载增强图像的逻辑
            filename = results['img_info']['filename']
            aug_filename = osp.join(results['aug_img_prefix'], filename)
            results['aug_img'] = mmcv.imread(aug_filename, self.color_type)
        return results

@PIPELINES.register_module()
class LoadAugmentedImageFromFile:
    """Load augmented image from file.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load augmented image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded augmented image and meta information.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['aug_img_prefix'] is not None:
            filename = osp.join(results['aug_img_prefix'],
                                results['img_info']['filename'])

            img_bytes = self.file_client.get(filename)
            aug_img = mmcv.imfrombytes(img_bytes, flag=self.color_type)

            if self.to_float32:
                aug_img = aug_img.astype(np.float32)

            # 调整增强图像尺寸以匹配原图
            if 'img_shape' in results:
                h, w = results['img_shape'][:2]
                aug_img = mmcv.imresize(aug_img, (w, h))

            results['aug_filename'] = filename
            results['aug_img'] = aug_img
            results['aug_img_shape'] = aug_img.shape
            results['aug_ori_shape'] = aug_img.shape
            if 'img_fields' not in results:
                results['img_fields'] = []
            results['img_fields'].append('aug_img')

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
import os
import pickle
import shutil
import sys

import cv2
from tqdm import tqdm


def create_all_dirs(path):
    if "." in path.split("/")[-1]:
        dirs = os.path.dirname(path)
    else:
        dirs = path
    os.makedirs(dirs, exist_ok=True)


def get_son_dir(dataroot):
    son_dir = []
    for _, dirs, _ in sorted(os.walk(dataroot)):
        for dname in sorted(dirs):
            son_dir.append(dname)
    return son_dir


def get_files_paths(dataroot, extensions, data_type='normal', include_son_dir=False):
    # '''
    # '''
    '''
    get file path list, support lmdb or normal files     lmdb暂时还没测试，不一定能用
    :param dataroot: 要输出文件路径的目录
    :param data_type: normal file  或者 lmdb
    :param extensions: 文件的扩展名
    :return:
    '''
    assert isinstance(extensions, list) or isinstance(extensions,
                                                      str), "extensions必须str或list, e.g: “png” or ['.jpg', '.JPG']"

    def _get_paths_from_lmdb(dataroot):
        '''get image path list from lmdb meta info'''
        meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
        paths = meta_info['keys']
        sizes = meta_info['resolution']
        if len(sizes) == 1:
            sizes = sizes * len(paths)
        return paths, sizes

    def _get_paths_from_normal(path, extensions):
        '''get file path list from file folder'''
        assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)

        def is_extension_file(filename, extensions):
            EXTENSIONS = []
            if isinstance(extensions, str):
                EXTENSIONS.append(extensions)
            else:
                EXTENSIONS = extensions
            return any(filename.endswith(extension) for extension in EXTENSIONS)

        flies = []
        deepth = 0
        for dirpath, dirs, fnames in sorted(os.walk(path)):
            if include_son_dir or deepth == 0:
                for fname in sorted(fnames):
                    if is_extension_file(fname, extensions):
                        img_path = os.path.join(dirpath, fname)
                        flies.append(img_path)
            deepth = 1
        if not flies:
            print('{:s} has no valid file'.format(path))
        return flies

    paths, sizes = None, None
    if data_type == 'lmdb':
        if dataroot is not None:
            paths, sizes = _get_paths_from_lmdb(dataroot)
        return paths, sizes
    elif data_type == 'normal':
        if dataroot is not None:
            paths = sorted(_get_paths_from_normal(dataroot, extensions))
        return paths
    else:
        raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))


def get_file_name(path, with_extension=True):
    if with_extension:
        return os.path.basename(path)
    else:
        return os.path.basename(path).split('.')[0]


def changeSR_name(HR_path, SR_path):
    HR_files = os.listdir(HR_path)
    SR_files = os.listdir(SR_path)
    HR_files.sort()
    SR_files.sort()
    for item_HR, item_SR in zip(HR_files, SR_files):
        # shotname, _ = os.path.splitext(HR_files)
        os.rename(os.path.join(SR_path, item_SR), os.path.join(SR_path, item_HR))


def changeSR_size(HR_path, SR_path):
    HR_files = os.listdir(HR_path)
    SR_files = os.listdir(SR_path)
    HR_files.sort()
    SR_files.sort()
    for item_HR, item_SR in zip(HR_files, SR_files):
        img_HR = cv2.imread(os.path.join(HR_path, item_HR))
        img_SR = cv2.imread(os.path.join(SR_path, item_SR))
        if img_HR.shape != img_SR.shape:
            img_SR = cv2.resize(img_SR, (img_HR.shape[1], img_HR.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(SR_path, item_SR), img_SR)
            # print(item_SR)

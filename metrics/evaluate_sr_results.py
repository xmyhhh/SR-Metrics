import os
import cv2
import math
import logging
import datetime
import pandas as pd
from PIL import Image
import LPIPS as models
import matlab.engine
import torch
import argparse
from tqdm import tqdm
from logging import handlers
import numpy as np
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
from tools.file import changeSR_name, get_files_paths, get_file_name


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def CalMATLAB(SR_path, GT_path, image_name, RGB2YCbCr):
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(eng.fullfile(os.getcwd(), 'MetricEvaluation')))
    res = eng.evaluate_results(SR_path, GT_path, image_name, RGB2YCbCr)
    res = np.array(res)
    res = res.squeeze()
    return res


def CalLPIPS(SR_path, GT_path):
    model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=False)

    imageA = np.array(Image.open(SR_path))
    imageB = np.array(Image.open(GT_path))
    imageA = torch.Tensor((imageA / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
    imageB = torch.Tensor((imageB / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
    dist = model.forward(imageA, imageB).detach().squeeze().numpy()

    res = np.array(dist)
    res = res.squeeze()
    return np.mean(res)


def evaluate_job(SR_path, HR_path, image_name, RGB2YCbCr):
    MATLAB = CalMATLAB(SR_path, HR_path, image_name, RGB2YCbCr)
    LPIPS = CalLPIPS(SR_path, HR_path)

    return (image_name, MATLAB, LPIPS)



parser = argparse.ArgumentParser(description="Evaluate SR results")
parser.add_argument('YAML', type=str, help='configuration file')
args = parser.parse_args()

conf = dict()
with open(args.YAML, 'r', encoding='UTF-8') as f:
    conf = yaml.load(f.read(), Loader=yaml.FullLoader)

Datasets = conf['Pairs']['Dataset']
SRFolder = conf['Pairs']['SRFolder']
GTFolder = conf['Pairs']['GTFolder']
RGB2YCbCr = conf['RGB2YCbCr']
max_workers = conf['max_workers']
Metric = ['Ma', 'NIQE', 'PI', 'PSNR', 'SSIM', 'MSE', 'RMSE', 'BRISQUE', 'LPIPS']
Name = conf['Name']
Echo = conf['Echo']

output_path = Name
xlsx_path = os.path.join('../evaluate', output_path, Name + '.xlsx')
if not os.path.isdir('../evaluate'):
    os.mkdir('../evaluate')

os.makedirs(os.path.join('../evaluate', output_path), exist_ok=True)

log = Logger(os.path.join('../evaluate', output_path, Name + '.log'), level='info')

log.logger.info('Init...')
log.logger.info('SRFolder - ' + str(Datasets))
log.logger.info('GTFolder - ' + str(GTFolder))
log.logger.info('SRFolder - ' + str(SRFolder))
log.logger.info('Metric - ' + str(Metric))
log.logger.info('Name - ' + Name)
log.logger.info('Echo - ' + str(Echo))

if os.path.exists(xlsx_path):
    res = pd.read_excel(xlsx_path, dtype=str)
else:
    res = pd.DataFrame(columns=('Dataset', 'PI', 'Ma', 'NIQE', 'MSE', 'RMSE', 'PSNR', 'SSIM', 'BRISQUE', 'LPIPS'))

for i, j, k in zip(Datasets, SRFolder, GTFolder):
    if i in res['Dataset'].unique():
        print("Evaluation of Dataset" + i + " already exist, pass")
    else:
        log.logger.info('Calculating ' + i + '...')
        if not set(os.listdir(j)) == set(os.listdir(k)):
            'SR pictures and GT pictures are not matched.'
            print("file name not same" + "HR_path:" + k + " SR_path" + j)
            changeSR_name(k, j)

        HR_paths = get_files_paths(k, extensions=['jpg', 'png'])
        SR_paths = get_files_paths(j, extensions=['jpg', 'png'])
        os.makedirs(os.path.join('../evaluate', output_path, "detail"), exist_ok=True)
        xlsx_detail_path = os.path.join('../evaluate', output_path, "detail", i + '.xlsx')
        if os.path.exists(xlsx_detail_path):
            res_detail = pd.read_excel(xlsx_detail_path, dtype=str)
        else:
            res_detail = pd.DataFrame(
                columns=('Name', 'PI', 'Ma', 'NIQE', 'MSE', 'RMSE', 'PSNR', 'SSIM', 'BRISQUE', 'LPIPS'))

        # new ThreadPool
        executor = ThreadPoolExecutor(max_workers=max_workers)
        all_task = []
        lock = Lock()
        for HR_path, SR_path in zip(HR_paths, SR_paths):
            image_name = get_file_name(HR_path, False)

            res_detail['Name'] = res_detail['Name'].astype('str')
            if image_name in res_detail['Name'].unique():
                print("Evaluation of Image" + image_name + " already exist, pass")

            else:
                # MATLAB = CalMATLAB(SR_path, HR_path, image_name, RGB2YCbCr)
                # LPIPS = CalLPIPS(SR_path, HR_path)

                args = [SR_path, HR_path, image_name, RGB2YCbCr]
                all_task.append(executor.submit(lambda p: evaluate_job(*p), args))

        for future in as_completed(all_task):
            image_name, MATLAB, LPIPS = future.result()

            resDict = dict()
            resDict['Name'] = [image_name]
            resDict['PI'] = [MATLAB[0]]
            resDict['Ma'] = [MATLAB[1]]
            resDict['NIQE'] = [MATLAB[2]]
            resDict['MSE'] = [MATLAB[3]]
            resDict['RMSE'] = [MATLAB[4]]
            resDict['PSNR'] = [MATLAB[5]]
            resDict['SSIM'] = [MATLAB[6]]
            resDict['BRISQUE'] = [MATLAB[7]]
            resDict['LPIPS'] = [LPIPS]
            resDataFrame = pd.DataFrame(resDict)
            res_detail = pd.concat([res_detail, resDataFrame])
            with lock:
                res_detail.to_excel(os.path.join('../evaluate', output_path, "detail", i + '.xlsx'), header=True,
                                    index=False)

        resDict = dict()
        resDict['Dataset'] = [i]
        resDict['PI'] = res_detail["PI"].mean()
        resDict['Ma'] = res_detail["Ma"].mean()
        resDict['NIQE'] = res_detail["NIQE"].mean()
        resDict['MSE'] = res_detail["MSE"].mean()
        resDict['RMSE'] = res_detail["RMSE"].mean()
        resDict['PSNR'] = res_detail["PSNR"].mean()
        resDict['SSIM'] = res_detail["SSIM"].mean()
        resDict['BRISQUE'] = res_detail["BRISQUE"].mean()
        resDict['LPIPS'] = res_detail["LPIPS"].mean()
        resDataFrame = pd.DataFrame(resDict)
        res = res.append(resDataFrame)
        if Echo:
            log.logger.info('[' + i + ']    Dataset - ' + str(resDict['Dataset']))
            log.logger.info('[' + i + ']    PI - ' + str(resDict['PI']))
            log.logger.info('[' + i + ']    Ma - ' + str(resDict['Ma']))
            log.logger.info('[' + i + ']  NIQE - ' + str(resDict['NIQE']))
            log.logger.info('[' + i + ']   MSE - ' + str(resDict['MSE']))
            log.logger.info('[' + i + ']  RMSE - ' + str(resDict['RMSE']))
            log.logger.info('[' + i + ']  PSNR - ' + str(resDict['PSNR']))
            log.logger.info('[' + i + ']  SSIM - ' + str(resDict['SSIM']))
            log.logger.info('[' + i + ']  BRISQUE - ' + str(resDict['BRISQUE']))
            log.logger.info('[' + i + ']  LPIPS - ' + str(resDict['LPIPS']))

# res.to_csv(os.path.join('../evaluate', output, Name + '.csv'), header=True, index=True)
res.to_excel(os.path.join('../evaluate', output_path, Name + '.xlsx'), header=True, index=False)

log.logger.info('Done.')

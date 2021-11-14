import os
import yaml
import numpy as np


def dict2Yaml(fileName, dataDict):
    stream = open(fileName, 'w', encoding='UTF-8')
    yaml.dump(dataDict, stream=stream, default_flow_style=False)
    conf = dict()


MethodDict = dict()
MethodDict['HCFlow_heat_a=23_b=5_t=0.95'] = ['DIV2K']
RGB2YCbCr = False
evaluate_Ma = False
max_workers = 8

bashFile = []

for method in MethodDict.keys():
    fileName = method + '.yml'
    dataDict = dict()
    dataDict['Pairs'] = dict()
    dataDict['Pairs']['Dataset'] = MethodDict[method]
    dataDict['Pairs']['SRFolder'] = []
    dataDict['Pairs']['GTFolder'] = []

    dataDict['RGB2YCbCr'] = RGB2YCbCr
    dataDict['evaluate_Ma'] = evaluate_Ma
    dataDict['max_workers'] = max_workers
    dataDict['Name'] = fileName
    dataDict['Echo'] = True
    for dataset in MethodDict[method]:
        dataDict['Pairs']['SRFolder'].append(str(os.path.join('../data', 'SR', dataset, method)))
        dataDict['Pairs']['GTFolder'].append(str(os.path.join('../data', 'GT', dataset, 'HR')))
    # bashFile.append('python evaluate_sr_results.py ' + str(fileName))
    dict2Yaml(fileName, dataDict)

print('Done.')

"""
batchAnalysis.py 

Code to anlayse batch sim results

Contributors: salvadordura@gmail.com
"""

import utils
import json, pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sb
import os
import collections
plt.style.use('seaborn-whitegrid')


def loadPlot(dataFolder, batchLabel, simLabel=None, include=None, timeRange=[0,1000], copyFiles=False, ext='.json'):
    from os import listdir
    from os.path import isfile, join
    
    jsonFolder = batchLabel 
    path = dataFolder+batchLabel #+'/noDepol'
    onlyFiles = [f for f in listdir(path) if isfile(join(path, f))
                and not f.endswith('batch.json')
                and not f.endswith('cfg.json')
                and not isfile(join(path, f[:-5] + '_traces.png'))]  # TEMPORARY - to avoid replacing existing ones !!

    if type(simLabel) is list:
        outfiles = [f for f in onlyFiles if any([f.endswith(sl+ext) for sl in simLabel])] 
    elif type(simLabel) is '':
        outfiles = [f for f in onlyFiles if f.endswith(simLabel+ext)]
    else:
        outfiles = [f for f in onlyFiles if f.endswith(ext) ] 
        
    if not include:
        allpops = ['IT2','PV2','SOM2','IT4','IT5A','PV5A','SOM5A','IT5B','PT5B','PV5B','SOM5B','IT6','CT6','PV6','SOM6']
        excpops = ['IT2','IT4','IT5A','IT5B','PT5B','IT6','CT6']
        inhpops = ['SOM2','PV2', 'SOM5A','PV5A',  'SOM5B','PV5B',  'SOM6', 'PV6']
        excpopsu = ['IT2','IT4','IT5A','PT5B']
        recordedTraces = [(pop,50) for pop in ['IT2', 'IT4', 'IT5A', 'PT5B']]+[('PT5B',x) for x in [393,447,579,19,104,214,1138,979,799]]

        include = {}
        include['traces'] = [(pop,0) for pop in allpops]
        include['traces'] = recordedTraces # [('IT2',50), ('IT5A',50), ('PT5B', 50)]
        include['raster'] = allpops # ['IT2', 'IT5A', 'PT5B'] #allpops +['S2','M2']
        include['hist'] = ['IT2', 'IT4', 'IT5A', 'IT5B', 'PT5B']
        include['stats'] = inhpops #inhpops #allpops #excpopsu
        include['rates'] = ['IT5A', 'PT5B']
        include['syncs'] = ['IT5A', 'PT5B']
        include['psd'] = ['allCells', 'IT2', 'IT5A', 'PT5B'] #['allCells']+excpops


    #with open('../sim/cells/popColors.pkl', 'rb') as fileObj: popColors = pickle.load(fileObj)['popColors']

    for outfile in outfiles:
        filename = dataFolder+jsonFolder+'/'+outfile
        print(filename)
        sim,data,out=utils.plotsFromFile(filename, 
                raster=1, 
                stats=0, 
                rates=0,
                syncs=0,
                hist=0, 
                psd=0, 
                traces=0, 
                grang=0, 
                plotAll=0, 
                timeRange=timeRange, 
                include=include, 
                textTop='')#, 
                #popColors=popColors, 
                #orderBy=['pop','y'])

        if copyFiles:
            sourceFile1 = dataFolder+jsonFolder+'/'+outfile.split('.json')[0]+'_traces_gid_3136.png'
            sourceFile2 = dataFolder+jsonFolder+'/'+outfile.split('.json')[0]+'_traces_gid_5198.png'
            cpcmd = 'cp ' + sourceFile1 + ' ' + path + '/.'
            cpcmd = cpcmd + '; cp ' + sourceFile2 + ' ' + path + '/.'
            os.system(cpcmd) 
            print(cpcmd)

    return sim,data


def plotConnFile(dataFolder, batchLabel):
    from netpyne import sim

    combs = [0, 1, 2]
    post = ['IT4', 'IT5A', 'IT5B', 'IT5B', 'PT5B']
    pre = ['PV5A', 'PV5B', 'SOM5A', 'SOM5B'] ## include IT2 ?
    feature = 'strength'
    graphType = 'bar'

    for i in combs:
        connsFile = dataFolder+batchLabel+'/%s_%d_conns_full.json'%(batchLabel, i)
        tagsFile = dataFolder+batchLabel+'/%s_%d_tags_full.json'%(batchLabel, i)
        saveFig = dataFolder+batchLabel+'/%s_%d_conn_%s_%s.png'%(batchLabel, i, feature, post[0])
        sim.analysis.plotConn(
            includePost=post, 
            includePre=pre,
            tagsFile=tagsFile, 
            connsFile=connsFile, 
            saveFig=saveFig,
            feature=feature,
            graphType=graphType)


def plot2DnetFile(dataFolder, batchLabel):
    from netpyne import sim

    combs = [0]
    include = ['PV2', 'SOM2', 'PV5A', 'PV5B', 'SOM5A', 'SOM5B', 'PV6', 'SOM6'] ## include IT2 ?
    include = ['IT2', 'IT4', 'IT5A', 'IT5B', 'PT5B', 'IT6', 'CT6']

    for i in combs:
        tagsFile = dataFolder+batchLabel+'/%s_%d_tags_fullpos.json'%(batchLabel, i)
        saveFig = dataFolder+batchLabel+'/%s_%d_2Dnet_%s.png'%(batchLabel, i, include[0])
        sim.analysis.plot2Dnet(
            include=include, 
            tagsFile=tagsFile, 
            saveFig=saveFig)


def plotConnDisynFile(dataFolder, batchLabel):
    from netpyne import sim

    combs = [0, 1, 2]
    pre = ['PV5A', 'PV5B', 'SOM5A', 'SOM5B'] 
    post = ['IT4', 'IT5A', 'IT5B', 'PT5B']

    for i in combs:
        jsonFile = dataFolder+batchLabel+'/%s_%d.json'%(batchLabel, i)
        sim.load(jsonFile)
        sim.cfg.compactConnFormat = False 
        sim.analysis.calculateDisynaptic(**{'includePost': post, 'includePre': pre, 'includePrePre': post})
        sim.analysis.plotConn(**{'showFig': False, 'saveFig': True,
                                        'includePost': post, 
                                        'includePre': pre,
                                        'feature': 'strength',
                                        'graphType': 'bar'})

    




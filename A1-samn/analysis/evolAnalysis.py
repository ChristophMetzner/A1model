# analyzeEvol.py 

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv
import json
import seaborn as sns
import IPython as ipy
import os
import utils
from pprint import pprint

def getParamLabels(dataFolder, batchSim):
    # get param labels
    with open('%s/%s/%s_batch.json' % (dataFolder, batchSim, batchSim), 'r') as f: 
        paramLabels = [str(x['label'][0])+str(x['label'][1]) if isinstance(x['label'], list) else str(x['label']) for x in json.load(f)['batch']['params']]
    return paramLabels

def loadData(dataFolder, batchSim, paramLabels):
    with open('%s/%s/%s_stats.csv'% (dataFolder, batchSim, batchSim)) as f: 
        reader = csv.reader(f)
        dfGens = pd.DataFrame(
                [{'gen': int(row[0]),
                'worst': float(row[2]),
                'best': float(row[3]),
                'avg': float(row[4]),
                'std': float(row[6])}
                for row in reader if reader.line_num > 1])

    with open('%s/%s/%s_stats_indiv.csv'% (dataFolder, batchSim, batchSim)) as f: 
        reader = csv.reader(f)
        dfParams = pd.DataFrame(
                [{**{'gen': int(row[0]),
                'cand': int(row[1]),
                'fit': float(row[2])},
                **{k: v for k,v in zip(paramLabels, [float(row[i].replace("[", "").replace("]", "")) for i in range(3, len(row))])}}
                for row in reader if reader.line_num > 1])

    with open('%s/%s/screenlog.0' % (dataFolder, batchSim)) as f:
        fits = []
        gen = 0
        for line in f:
            if line.strip().startswith('Waiting for jobs from generation'):
                gen = line.split('Waiting for jobs from generation ')[1].split('/')[0]
            if line.strip().startswith('IT2'):
                nextLine = f.readline()
                cand = nextLine.split()[1]
                avgFit = nextLine.split()[4]
                data = line.strip().split()
                popFit = [{'gen_cand': '%d_%d' %(int(gen),int(cand)), 'gen': int(gen), 'cand': int(cand), 'avgFit': float(avgFit), 'pop': data[i], 'rate': float(data[i+1].split('=')[1]), 'fit': float(data[i+2].split('=')[1].replace(';' ,''))} for i in range(0, len(data), 3)]
                fits.extend(popFit)
            if line.startswith('There was an exception evaluating candidate'):
                cand = int(line.split('candidate ')[1].split(':')[0])
                popFit = [{'gen_cand':'%d_%d' % (int(gen), int(cand)), 'gen':int(gen), 'cand':int(cand), 'avgFit':1000, 'pop':data[i], 'rate':-1, 'fit':-1} for i in range(0, len(data), 3)]
                fits.extend(popFit)

        dfPops = pd.DataFrame(fits)

    return dfGens, dfParams, dfPops

def plotFitnessEvol(dataFolder, batchSim, df):
    df = df.drop(['gen', 'avg', 'std', 'worst'], axis=1)
    df.plot(figsize=(12, 8))
    plt.xlabel('Generation')
    plt.ylabel('Fitness error')
    plt.savefig('%s/%s/%s_fitness.png' % (dataFolder, batchSim, batchSim))

def plotParamsVsFitness(dataFolder, batchSim, df, paramLabels, excludeAbove=None, ylim=None):

    if excludeAbove:
        df = df[df.fit < excludeAbove]

    df2 = df.drop(['gen', 'cand', 'fit'], axis=1)
    fits = list(df['fit'])
    plt.figure(figsize=(16,12))
    for i, (k,v) in enumerate(df2.items()):
        y = v #(np.array(v)-min(v))/(max(v)-min(v)) # normalize
        x = np.random.normal(i, 0.04, size=len(y))         # Add some random "jitter" to the x-axis
        s = plt.scatter(x, y, alpha=0.3, c=[int(f-1) for f in fits], cmap='jet_r')
    plt.colorbar(label = 'fitness')
    plt.ylabel('Parameter value')
    plt.xlabel('Parameter')
    plt.xticks(range(len(paramLabels)), paramLabels, rotation=45)
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95)
    if ylim: plt.ylim(0, ylim)
    plt.savefig('%s/%s/%s_scatter_params_%s.png' % (dataFolder, batchSim, batchSim, 'excludeAbove-'+str(excludeAbove) if excludeAbove else ''))
    #plt.show()

def plotParamsVsFitness2D(dataFolder, batchSim, df, paramLabels):
    plt.rcParams.update({'font.size': 12})
    
    df2 = df.drop(['fit', 'cand', 'gen'], axis=1)
    df2Norm = normalize(df2)

    import seaborn as sns
    pp = sns.pairplot(df2Norm, size=1.5, aspect=1.5, markers='o',
                    #hue='gen',
                    plot_kws=dict(s=10),#, edgecolor="k", linewidth=0.5),
                    diag_kind="kde", diag_kws=dict(shade=True))    


    plt.savefig('%s/%s/%s_scatter2d_params.png' % (dataFolder, batchSim, batchSim))

def plotRatesVsFitness(dataFolder, batchSim, df, ymax=None):
    df2 = df.drop(['gen_cand', 'fit'], axis=1)
    df3 = df2.groupby('pop')
    pops = df3.groups.keys()

    plt.figure(figsize=(16,12))
    for i, (k,v) in enumerate(df3):
        y = list(v['rate'].values)  #(np.array(v)-min(v))/(max(v)-min(v)) # normalize
        fits = list(v['avgFit'].values)
        x = np.random.normal(i, 0.04, size=len(y))         # Add some random "jitter" to the x-axis
        s = plt.scatter(x, y, alpha=0.3, c=[int(f-1) for f in fits], cmap='jet_r')
    plt.colorbar(label = 'candidate fitness')
    plt.ylabel('firing rate')
    if ymax: plt.ylim(0,ymax)
    plt.xlabel('population')
    plt.xticks(range(len(pops)), pops, rotation=45)
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95)
    plt.savefig('%s/%s/%s_scatter_pops_%s.png' % (dataFolder, batchSim, batchSim, str(ymax) if ymax else ''))
    #plt.show()

def plotParamsVsRates(dataFolder, batchsim, dfParams, dfPops, pops, excludeAbove):

    if excludeAbove:
        dfParams = dfParams[dfParams.fit < excludeAbove]

    dfMerge = pd.merge(dfParams, dfPops, on=['gen', 'cand'])
    dfMerge = dfMerge.query('rate>=0.0')

    for pop in pops:

        popRates = list(dfMerge.query('pop==@pop').sort_values(by=['gen', 'cand']).rate)
        dfMergeParamsPop1 = dfMerge.query('pop==@pop').sort_values(by=['gen', 'cand'])
        dfMergeParamsPop2 = dfMergeParamsPop1.drop(['gen', 'cand', 'gen_cand', 'fit_x', 'fit_y', 'rate', 'avgFit', 'pop'], axis=1)
        
        plt.figure(figsize=(16,12))
        for i, (k,v) in enumerate(dfMergeParamsPop2.items()):
            y = (np.array(v)-min(v))/(max(v)-min(v)) # normalize
            x = np.random.normal(i, 0.10, size=len(y))         # Add some random "jitter" to the x-axis
            s = plt.scatter(x, y, alpha=0.1, c=popRates, cmap='jet_r')
        plt.colorbar(label = '%s rate (Hz)' % pop)
        plt.ylabel('normalized parameter value')
        plt.xlabel('parameter')
        plt.xticks(range(len(paramLabels)), paramLabels, rotation=45)
        plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95)
        plt.savefig('%s/%s/%s_scatter_params_pop_%s_%s.png' % (dataFolder, batchSim, batchSim, pop,'excludeAbove-'+str(excludeAbove) if excludeAbove else ''))

        #ipy.embed()

        #plt.show()

def plotRatesVsParams(dataFolder, batchsim, dfParams, dfPops, ymax, excludeAbove, excludeBelow):

    if excludeAbove:
        dfParams = dfParams[dfParams.fit < excludeAbove]

    if excludeBelow:
        dfParams = dfParams[dfParams.fit < excludeBelow]

    dfParams = dfParams.rename(columns={'IELayerGain1-3': 'IELayerGain13', 'IILayerGain1-3': 'IILayerGain13'})
    dfMerge = pd.merge(dfParams, dfPops, on=['gen', 'cand'])
    dfMerge = dfMerge.query('rate>=0.0')
    dfMerge2 = dfMerge.drop(['gen_cand', 'cand', 'gen', 'avgFit', 'fit_x', 'fit_y'], axis=1)

    for param in [x for x in dfMerge2.columns if x not in ['pop', 'rate']]:
        print('Plotting scatter of rate vs %s param ...' %(param))
        plt.figure(figsize=(16, 12))
        dfMerge3 = dfMerge2.query('%s >= 0.0' % (param))
        dfMerge4 = dfMerge3.groupby('pop')
        pops = dfMerge4.groups.keys()
        for i, (k,v) in enumerate(dfMerge4):
            y = list(v['rate'].values)  #(np.array(v)-min(v))/(max(v)-min(v)) # normalize
            vals = list(v[param].values)
            x = np.random.normal(i, 0.04, size=len(y))         # Add some random "jitter" to the x-axis
            s = plt.scatter(x, y, alpha=0.1, c=vals, cmap='jet_r')
        plt.colorbar(label = 'Parameter value')
        plt.ylabel('firing rate')
        if ymax: plt.ylim(0,ymax)
        plt.xlabel('population')
        plt.xticks(range(len(pops)), pops, rotation=45)
        plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95)
        plt.title('Parameter: %s' % (param))
        plt.savefig('%s/%s/%s_scatter_rates_%s_%s_%s_%s.png' %
            (dataFolder, batchSim, batchSim, param, str(ymax) if ymax else '', 'excludeBelow-'+str(excludeBelow) if excludeBelow else '', 'excludeAbove-'+str(excludeAbove) if excludeAbove else ''))


        #ipy.embed()

        #plt.show()

def normalize(df, exclude=[]):
    result = df.copy()
    for feature_name in [f for f in df.columns if f not in exclude]:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def filterRates(df, condlist=['rates', 'I>E', 'E5>E6>E2', 'PV>SOM'], copyFolder=None, dataFolder=None, batchLabel=None, skipDepol=False):
    from os.path import isfile, join
    from glob import glob

    df = df[['gen_cand', 'pop', 'rate']].pivot(columns='pop', index='gen_cand')
    df.columns = df.columns.droplevel(0)

    allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']

    ranges = {}
    Erange = [0.05,100]
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B', 'PT5B', 'IT6','CT6', 'TC', 'TCM', 'HTC']
    for pop in Epops:
        ranges[pop] = Erange
    
    conds = []
    # check pop rate ranges
    if 'rates' in condlist:
        for k,v in ranges.items(): conds.append(str(v[0]) + '<=' + k + '<=' + str(v[1]))
    condStr = ''.join([''.join(str(cond) + ' and ') for cond in conds])[:-4]
    dfcond = df.query(condStr)

    ranges = {}
    Irange = [0.05,180]
    Ipops = ['NGF1',                        # L1
        'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
        'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
        'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
        'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
        'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',#,  # L5B
        'SOM6', 'VIP6', 'NGF6',
        'IRE', 'IREM', 'TI']      # L6 PV6

    for pop in Ipops:
        ranges[pop] = Irange

    conds = []

    # check pop rate ranges
    if 'rates' in condlist:
        for k,v in ranges.items(): conds.append(str(v[0]) + '<=' + k + '<=' + str(v[1]))
    condStr = ''.join([''.join(str(cond) + ' and ') for cond in conds])[:-4]
    dfcond = dfcond.query(condStr)


    # # check I > E in each layer
    # if 'I>E' in condlist:
    #     conds.append('PV2 > IT2 and SOM2 > IT2')
    #     conds.append('PV5A > IT5A and SOM5A > IT5A')
    #     conds.append('PV5B > IT5B and SOM5B > IT5B')
    #     conds.append('PV6 > IT6 and SOM6 > IT6')

    # # check E L5 > L6 > L2
    # if 'E5>E6>E2' in condlist:
    #     #conds.append('(IT5A+IT5B+PT5B)/3 > (IT6+CT6)/2 > IT2')
    #     conds.append('(IT5A+IT5B+PT5B)/3 > (IT6+CT6)/2')
    #     conds.append('(IT6+CT6)/2 > IT2')
    #     conds.append('(IT5A+IT5B+PT5B)/3 > IT2')
    
    # # check PV > SOM in each layer
    # if 'PV>SOM' in condlist:
    #     conds.append('PV2 > IT2')
    #     conds.append('PV5A > SOM5A')
    #     conds.append('PV5B > SOM5B')
    #     conds.append('PV6 > SOM6')


    # construct query and apply
    # condStr = ''.join([''.join(str(cond) + ' and ') for cond in conds])[:-4]
    # dfcond = df.query(condStr)

    print('\n Filtering based on: ' + str(condlist) + '\n' + condStr)
    print(dfcond)
    print(len(dfcond))

    # copy files
    if copyFolder:
        targetFolder = dataFolder+batchLabel+'/'+copyFolder
        try: 
            os.mkdir(targetFolder)
        except:
            pass
        
        for i,row in dfcond.iterrows():     
            if skipDepol:
                sourceFile1 = dataFolder+batchLabel+'/noDepol/'+batchLabel+row['simLabel']+'*.png'  
            else:
                sourceFile1 = dataFolder+batchLabel+'/gen_'+i.split('_')[0]+'/gen_'+i.split('_')[0]+'_cand_'+i.split('_')[1]+'_*raster*.png'   
            #sourceFile2 = dataFolder+batchLabel+'/'+batchLabel+row['simLabel']+'.json'
            if len(glob(sourceFile1))>0:
                cpcmd = 'cp ' + sourceFile1 + ' ' + targetFolder + '/.'
                #cpcmd = cpcmd + '; cp ' + sourceFile2 + ' ' + targetFolder + '/.'
                os.system(cpcmd) 
                print(cpcmd)


    return dfcond

def testFitness(file, timeRange):
    allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 
    'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
    'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'IC'] 
    
    sim = utils.loadFromFile(file, allpops)

    spkts, spkids = zip(*[(t,i) for t,i in zip(sim.allSimData['spkt'], sim.allSimData['spkid']) if timeRange[0] <= t <= timeRange[1]])
    
    isicv = {}
    for pop in allpops:
        try:
            spkt = [t for t, i in zip(spkts, spkids) if i in sim.net.pops[pop].cellGids]
            spkt.insert(0, timeRange[0])
            spkt.append(timeRange[1])
            isimat = [t - s for s, t in zip(spkt, spkt[1:])]
            isicv[pop] = np.std(isimat) / np.mean(isimat)
            if np.isnan(isicv[pop]):
                print('Not enough spikes for pop %s; setting to 50' % (pop))
                print(pop, spkt, isimat, isicv[pop])
                isicv[pop] = 50
        except:
            print('Exception processing pop %s' % (pop))
            #print(pop,spkt,isimat,isicv)

    for pop in allpops:
        try:
            print('%s: %.2f' % (pop, isicv[pop]))
        except:
            pass

    return isicv


def testFitness2(file, timeRange):
    allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 
    'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B',
    'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'IC'] 
    
    sim = utils.loadFromFile(file, allpops)
    
    tranges = [[500, 750], [750, 1000], [1000, 1250], [1250, 1500]]
    sim.allSimData['popRates'] = sim.analysis.popAvgRates(tranges)

    simData = sim.allSimData
    
    #fitness func
    fitnessFuncArgs = {}
    pops = {}
    
    ## Exc pops
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'PT5B', 'CT5B', 'IT6', 'CT6', 'TC', 'TCM', 'HTC']  # all layers + thal + IC

    Etune = {'target': 5, 'width': 20, 'min': 0.05}
    for pop in Epops:
        pops[pop] = Etune
    
    ## Inh pops 
    Ipops = ['NGF1',                            # L1
            'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
            'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
            'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
            'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
            'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',  # L5B
            'PV6', 'SOM6', 'VIP6', 'NGF6',       # L6
            'IRE', 'IREM', 'TI']  # Thal 

    Itune = {'target': 10, 'width': 30, 'min': 0.05}
    for pop in Ipops:
        pops[pop] = Itune
    
    
    maxFitness = 1000
    popFitnessAll = []

    for trange in tranges:
        popFitnessAll.append([min(np.exp(abs(v['target'] - simData['popRates'][k][(trange[0], trange[1])])/v['width']), maxFitness) 
            if simData['popRates'][k][(trange[0], trange[1])] > v['min'] else maxFitness for k, v in pops.items()])
    
    popFitness = np.mean(np.array(popFitnessAll), axis=0)
    fitness = np.mean(popFitness)

    popInfo = '; '.join(['%s rate=%.1f fit=%1.f' % (p, np.mean(list(simData['popRates'][p].values())), popFitness[i]) for i,p in enumerate(pops)])
    print('  ' + popInfo)
    print(fitness)
        
    return fitness


#fitness = testFitness2('../data/v23_batch10/gen_5/gen_5_cand_34.json', [500, 1500])
#isicv28 = testFitness2('../data/v23_batch10/gen_5/gen_5_cand_28.json', [500, 1500])



# -----------------------------------------------------------------------------
# Main code
# -----------------------------------------------------------------------------
if __name__ == '__main__': 
    dataFolder = '../data/'
    batchSim = 'v23_batch12' 

    allpops = ['NGF1', 'IT2', 'PV2', 'SOM2', 'VIP2', 'NGF2', 'IT3', 'SOM3', 'PV3', 'VIP3', 'NGF3', 'ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT5A', 'CT5A', 'PV5A', 'SOM5A', 'VIP5A', 'NGF5A', 'IT5B', 'PT5B', 'CT5B', 'PV5B', 'SOM5B', 'VIP5B', 'NGF5B', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI', 'IC']

    # set font size
    plt.rcParams.update({'font.size': 18})

    # get param labels
    paramLabels = getParamLabels(dataFolder, batchSim)

    # load evol data from files
    dfGens, dfParams, dfPops = loadData(dataFolder, batchSim, paramLabels)

    # # plot fitness evolution across generations
    # plotFitnessEvol(dataFolder, batchSim, dfGens)

    # # # # plot param dsitributions
    plotParamsVsFitness(dataFolder, batchSim, dfParams, paramLabels, excludeAbove=200, ylim=2.0)
    # plotParamsVsFitness2D(dataFolder, batchSim, dfParams, paramLabels)

    # #  plot pop fit dsitributions
    # NOTE: BUG!! gen and cand do not correspond in .csv and screenlog!!!!
    # plotRatesVsFitness(dataFolder, batchSim, dfPops)
    # plotRatesVsFitness(dataFolder, batchSim, dfPops, 50)

    #plotParamsVsRates(dataFolder, batchSim, dfParams, dfPops, allpops, excludeAbove=400)
    #plotRatesVsParams(dataFolder, batchSim, dfParams, dfPops, ymax=50, excludeAbove=400, excludeBelow=None)


    # filter results by pop rates
    #dfFilter = filterRates(dfPops, condlist=['rates'], copyFolder='best', dataFolder=dataFolder, batchLabel=batchSim, skipDepol=False) # ,, 'I>E', 'E5>E6>E2' 'PV>SOM']

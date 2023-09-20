# analyzeEvol.py 

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv
import json
import seaborn as sns

def getParamLabels(dataFolder, batchSim):
    # get param labels
    with open('%s/%s/%s_batch.json' % (dataFolder, batchSim, batchSim), 'r') as f: 
        paramLabels = [str(x['label'][1]) + '_' + str(x['label'][2]) + '_' + str(x['label'][3]) if isinstance(x['label'], list) and len(x['label'])==4
                     else str(x['label'][1]) + '_' + str(x['label'][2]) for x in json.load(f)['batch']['params']]
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
                'size': int(row[1]),
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
                popFit = [{'gen_cand': '%d_%d' %(int(gen),int(cand)), 'avgFit': float(avgFit), 'pop': data[i], 'rate': float(data[i+1].split('=')[1]), 'fit': float(data[i+2].split('=')[1].replace(';' ,''))} for i in range(0, len(data), 3)]
                fits.extend(popFit)
        dfPops = pd.DataFrame(fits)

    return dfGens, dfParams, dfPops


def plotParams(dataFolder, batchSim, df, paramLabels):

    df2 = df.drop(['gen', 'size', 'fit'], axis=1)
    fits = list(df['fit'])
    maxValue = 1000 

    plt.figure(figsize=(16,12))
    for i, (k,v) in enumerate(df2.items()):
        y = (np.array(v)-min(v))/(max(v)-min(v)) # normalize
        x = np.random.normal(i, 0.04, size=len(y))         # Add some random "jitter" to the x-axis
        s = plt.scatter(x, y, alpha=0.3, c=[int(f-1) for f in fits], cmap='jet_r')
    plt.colorbar(label = 'fitness')
    plt.ylabel('normalized parameter value')
    plt.xlabel('parameter')
    plt.xticks(range(len(paramLabels)), paramLabels, rotation=45)
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95)
    plt.savefig('%s/%s/%s_scatter_params.png' % (dataFolder, batchSim, batchSim))
    #plt.show()

def plotParams2D(dataFolder, batchSim, df, paramLabels):
    plt.rcParams.update({'font.size': 12})
    
    df2 = df.drop(['fit', 'size', 'gen'], axis=1)
    df2Norm = normalize(df2)

    import seaborn as sns
    pp = sns.pairplot(df2Norm, size=1.5, aspect=1.5, markers='o',
                    #hue='gen',
                    plot_kws=dict(s=10),#, edgecolor="k", linewidth=0.5),
                    diag_kind="kde", diag_kws=dict(shade=True))    


    plt.savefig('%s/%s/%s_scatter2d_params.png' % (dataFolder, batchSim, batchSim))

def plotPopRates(dataFolder, batchSim, df):
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
    plt.xlabel('population')
    plt.xticks(range(len(pops)), pops, rotation=45)
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95)
    plt.savefig('%s/%s/%s_scatter_pops.png' % (dataFolder, batchSim, batchSim))
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
    Erange = [0.1,1000]
    Epops = ['IT2', 'IT3', 'ITP4', 'ITS4', 'IT5A', 'CT5A', 'IT5B', 'CT5B', 'PT5B', 'IT6','CT6', 'TC', 'TCM', 'HTC', 'IRE', 'IREM', 'TI']
    for pop in Epops:
        ranges[pop] = Erange

    ranges = {}
    Irange = [0.1,1000]
    Ipops = ['NGF1',                        # L1
        'PV2', 'SOM2', 'VIP2', 'NGF2',      # L2
        'PV3', 'SOM3', 'VIP3', 'NGF3',      # L3
        'PV4', 'SOM4', 'VIP4', 'NGF4',      # L4
        'PV5A', 'SOM5A', 'VIP5A', 'NGF5A',  # L5A  
        'PV5B', 'SOM5B', 'VIP5B', 'NGF5B',#,  # L5B
        'SOM6', 'VIP6', 'NGF6']      # L6 PV6

    for pop in Ipops:
        ranges[pop] = Irange

    conds = []

    # check pop rate ranges
    if 'rates' in condlist:
        for k,v in ranges.items(): conds.append(str(v[0]) + '<=' + k + '<=' + str(v[1]))
    
    # check I > E in each layer
    if 'I>E' in condlist:
        conds.append('PV2 > IT2 and SOM2 > IT2')
        conds.append('PV5A > IT5A and SOM5A > IT5A')
        conds.append('PV5B > IT5B and SOM5B > IT5B')
        conds.append('PV6 > IT6 and SOM6 > IT6')

    # check E L5 > L6 > L2
    if 'E5>E6>E2' in condlist:
        #conds.append('(IT5A+IT5B+PT5B)/3 > (IT6+CT6)/2 > IT2')
        conds.append('(IT5A+IT5B+PT5B)/3 > (IT6+CT6)/2')
        conds.append('(IT6+CT6)/2 > IT2')
        conds.append('(IT5A+IT5B+PT5B)/3 > IT2')
    
    # check PV > SOM in each layer
    if 'PV>SOM' in condlist:
        conds.append('PV2 > IT2')
        conds.append('PV5A > SOM5A')
        conds.append('PV5B > SOM5B')
        conds.append('PV6 > SOM6')


    # construct query and apply
    condStr = ''.join([''.join(str(cond) + ' and ') for cond in conds])[:-4]
    
    dfcond = df.query(condStr)
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
                sourceFile1 = dataFolder+batchLabel+'/'+batchLabel+row['simLabel']+'*.png'   
            #sourceFile2 = dataFolder+batchLabel+'/'+batchLabel+row['simLabel']+'.json'
            if len(glob(sourceFile1))>0:
                cpcmd = 'cp ' + sourceFile1 + ' ' + targetFolder + '/.'
                #cpcmd = cpcmd + '; cp ' + sourceFile2 + ' ' + targetFolder + '/.'
                os.system(cpcmd) 
                print(cpcmd)


    return dfcond


# -----------------------------------------------------------------------------
# Main code
# -----------------------------------------------------------------------------
dataFolder = 'data/'
batchSim = 'NGF_evol' 

# set font size
plt.rcParams.update({'font.size': 14})

# get param labels
paramLabels = getParamLabels(dataFolder, batchSim)

# load evol data from files
dfGens, dfParams, dfPops = loadData(dataFolder, batchSim, paramLabels)

# plot param dsitributions
plotParams(dataFolder, batchSim, dfParams, paramLabels)
#plotParams2D(dataFolder, batchSim, dfParams, paramLabels)

# # # plot pop fit dsitributions
#plotPopRates(dataFolder, batchSim, dfPops)

# filter results by pop rates
#dfFilter = filterRates(dfPops, condlist=['rates'], copyFolder=None, dataFolder=None, batchLabel=None, skipDepol=False) # ,, 'I>E', 'E5>E6>E2' 'PV>SOM']

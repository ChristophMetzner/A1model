import json
import sys

fnjson = 'sim.json'

for i in range(len(sys.argv)):
  if sys.argv[i].endswith('.json'):
    fnjson = sys.argv[i]
    print('reading ', fnjson)

def readconf (fnjson):    
  with open(fnjson,'r') as fp:
    dconf = json.load(fp)
    #print(dconf)
  return dconf

def checkDefVal (d, k, val):
  # check if k is in d, if not, set d[k] = val
  if k not in d: d[k] = val

def ensureDefaults (dconf):
  # make sure (some of the) default values are present so dont have to check for them throughout rest of code
  checkDefVal(dconf,'net',{})
  checkDefVal(dconf, 'verbose', 0)

dconf = readconf(fnjson) # read the configuration
ensureDefaults(dconf) # ensure some of the default values present


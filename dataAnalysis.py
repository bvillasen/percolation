import numpy as np
import sys, time, os, inspect, datetime
import h5py as h5
import matplotlib.pyplot as plt

def plotConc( maxVals, sumConc, boundatySum, iterations ):
  plt. figure(0)
  plt.clf()
  plt.plot(iterations, sumConc, '--bo' )
  plt.title("Concentration Sum")
  ax = plt.gca()
  ax.set_xlabel("Time")
  
  plt.figure(1)
  plt.clf()
  plt.yscale("log")
  plt.plot(iterations, maxVals, '--bo')
  plt.title("Concentration Max Value")
  ax = plt.gca()
  ax.set_xlabel("Time")
  
  plt.figure(2)
  plt.clf()
  #plt.yscale("log")
  plt.plot(iterations, boundatySum, '--bo')
  plt.title("Boundary Sum")
  ax = plt.gca()
  ax.set_xlabel("Time")
  plt.draw()
  
def plotCM( cmX, cmY, iterations, p, h, plotY=False, notFirst=True, sqrtX=False  ):
  #wm = plt.get_current_fig_manager()
  #wm.window.wm_geometry("400x900+50+50")
  plt. figure(0)
  plt.clf()
  if sqrtX:
    iterArray = np.array(iterations)
    plt.plot( iterArray**2, cmX, '--bo' )
  else:  plt.plot(iterations[1*notFirst:], cmX[1*notFirst:], '--bo' )
  plt.title("CM_x  p={0:1.3f}  h={1:1.3f}".format(float(p), float(h)))
  ax = plt.gca()
  ax.set_xlabel("Time")
  
  if plotY:
    plt. figure(1)
    plt.clf()
    plt.plot(iterations[1*notFirst:], cmY[1*notFirst:], '--bo' )
    plt.title("CM_y")
    ax = plt.gca()
    ax.set_xlabel("Time")
  plt.draw()
  
def loadData(dataFileName, escapedMax=1e-4, removeEscaped=True):
  p = float( dataFileName[dataFileName.find("p_")+2:dataFileName.find("h_")] )/100
  h = float( dataFileName[dataFileName.find("h_")+2:dataFileName.find("H_")] )/100
  H = int( dataFileName[dataFileName.find("H_")+2:dataFileName.find("W_")] )
  W = int( dataFileName[dataFileName.find("W_")+2:dataFileName.find("R_")] )
  try: dataFile = h5.File( dataFileName ,'r')
  except IOError: 
    print "ERROR:  Unable to open file"
    return
  iterations = dataFile.get("iterations")[...]
  dataCM = dataFile.get("CM_data")[...]
  boundaryData = dataFile.get("boundary_data")[...]
  nRealiz = dataCM.shape[0]
  boundaryEscaped = boundaryData >= escapedMax
  nEscaped = sum( boundaryEscaped )
  escapedIndexes = list(boundaryEscaped.nonzero()[0].flat)
  goodIndexes = list((boundaryData < escapedMax).nonzero()[0].flat)
  if len(escapedIndexes)>0:
    #print "\nLoading Data: {2}\np = {0:1.2}   h = {1:1.2f}  nRealiz: {3} ". format( float(p), float(h), dataFileName, nRealiz )
    print "ESC: p={0:1.2f}   h={1:1.2f}   Boundary escaped: ( {2}, {3:1.4f} ) grid: ( {4} x {5} )".format( float(p), float(h), nEscaped, float(boundaryData.max()), H, W )
  return p, h, nRealiz, dataCM, iterations, escapedIndexes, goodIndexes 

  
def plotRealizations( dataFileName, escapedMax=1e-4, removeEscaped=True, nCurves=None ):
  p, h, nRealiz, dataCM, iterations, escapedIndexes, goodIndexes = loadData( dataFileName, escapedMax, removeEscaped )
  nEscaped = len( escapedIndexes )
  if not nCurves: nCurves=nRealiz
  plt. figure()
  #plt.clf()
  for i in range( nCurves ):
    if i in goodIndexes: plt.plot(iterations, dataCM[i][0], ) 
    if i in escapedIndexes: plt.plot(iterations, dataCM[i][0], '--r' )    
  plt.title("CM_x  p={0:1.2f}  h={1:1.2f}  Realiz={2}   removed={3}".format(float(p), float(h), nRealiz, nEscaped*removeEscaped) )
  ax = plt.gca()
  ax.set_xlabel("Time")
  plt.show()


def analizeRealizations( dataFileName, lLim=0, rLim=49, escapedMax=1e-4, removeEscaped=True ):
  p, h, nRealiz, dataCM, iterations, escapedIndexes, goodIndexes = loadData( dataFileName, escapedMax, removeEscaped )
  nEscaped = len( escapedIndexes )
  if removeEscaped:
    nGood = len( goodIndexes )
    startPointsX = dataCM[ goodIndexes, 0, lLim ]
    endPointsX = dataCM[ goodIndexes, 0, rLim ]
  else:
    nGood = nRealiz
    startPointsX = dataCM[ :, 0, leftLimit ]
    endPointsX = dataCM[ :, 0, rigthLimit ]
  deltaX = endPointsX - startPointsX
  velX = deltaX/(iterations[rLim] - iterations[lLim] )
  avrgVelX = sum( velX )/nGood
  stdvVelX = np.sqrt( sum( (velX-avrgVelX)*(velX-avrgVelX) ) )/nGood
  return [ h, avrgVelX, stdvVelX ]

def getVelXDistribution( p, leftLimit=0, rigthLimit=49, escapedMax=1e-4 ):
  lookForP = "p_{0:2.0f}".format( float(p*100) )
  lookForW = "W_{0}".format( 1024 )
  velX_data =[ analizeRealizations( dataDir + dataFileName, lLim=leftLimit, rLim=rigthLimit, escapedMax=escapedMax ) for dataFileName in allDataFiles if (dataFileName.find( lookForP )>=0) ]# and (dataFileName.find(lookForW)>=0)  ]
  velX_data = np.array( velX_data ).T
  return velX_data

def plotVelXDistribution( p, leftLimit=0, rigthLimit=49, escapedMax=1e-4 ):
  velX_data = getVelXDistribution( p, leftLimit, rigthLimit, escapedMax )
  plt.figure()
  plt.errorbar( velX_data[0], velX_data[1], yerr=velX_data[2], )
  plt.show()  
  
def plotVelX( p_list, leftLimit=0, rigthLimit=49, escapedMax=1e-4):
  plt.figure()
  for p in p_list:
    velX_data = getVelXDistribution( p, leftLimit, rigthLimit, escapedMax )
    plt.errorbar( velX_data[0], velX_data[1], yerr=velX_data[2], label="p={0:1.2f}".format(float(p)) )
  ax = plt.gca()
  ax.set_xlabel("$h$")
  ax.set_ylabel("$v_x$")
  plt.legend(prop={'size':8})
  plt.show()  
  


dataDir = "data/"
allDataFiles = os.listdir( dataDir )
allDataFiles.sort()

if __name__ == "__main__":
  #Read al data files in data direcotry
  #print "Data Files to anlize:"
  #for dataFile in allDataFiles:
    #print " ", dataFile

  plotVelXDistribution( 0.39, leftLimit=0, rigthLimit=49, escapedMax=1e-4  )
  












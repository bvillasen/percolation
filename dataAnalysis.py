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
  
  
def plotRealizations( dataFileName, escapedMax=1e-4, removeEscaped=True ):
  p = float( dataFileName[dataFileName.find("p_")+2:dataFileName.find("h_")] )/100
  h = float( dataFileName[dataFileName.find("h_")+2:dataFileName.find("H_")] )/100
  print "Loading Data...\n p = {0:1.2}\n h = {1:1.2f}". format( float(p), float(h) )
  dataFile = h5.File( dataFileName ,'r')
  iterations = dataFile.get("iterations")[...]
  dataCM = dataFile.get("CM_data")[...]
  boundaryData = dataFile.get("boundary_data")[...]
  nRealiz = dataCM.shape[0]
  boundaryEscaped = boundaryData >= escapedMax
  nEscaped = sum( boundaryEscaped )
  escapedIndexes = set(boundaryEscaped.nonzero()[0].flat)
  goodIndexes = set((boundaryData < escapedMax).nonzero()[0].flat)
  print " nRealiz: {0}\n  Boundary escaped: {1}\n".format( nRealiz, nEscaped )
  #print goodIndexes
  #print escapedIndexes
  plt. figure()
  #plt.clf()
  for i in range( nRealiz ):
    if i in goodIndexes: plt.plot(iterations, dataCM[i][0], ) 
    if i in escapedIndexes: plt.plot(iterations, dataCM[i][0], '--r' )    
  plt.title("CM_x  p={0:1.2f}  h={1:1.2f}  Realiz={2}   removed={3}".format(float(p), float(h), nRealiz, nEscaped*removeEscaped) )
  ax = plt.gca()
  ax.set_xlabel("Time")
  dataFile.close()
  plt.show
  return dataCM
  
import numpy as np
import sys, time, os, inspect, datetime
#import h5py as h5
import matplotlib.pyplot as plt

def plotData( maxVals, sumConc, iterations ):
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
  plt.draw()
  
def plotCM( cmX, cmY, iterations, plotY=False, notFirst=True, sqrtX=False  ):
  #wm = plt.get_current_fig_manager()
  #wm.window.wm_geometry("400x900+50+50")
  plt. figure(0)
  plt.clf()
  if sqrtX:
    iterArray = np.array(iterations)
    plt.plot( iterArray**2, cmX, '--bo' )
  else:  plt.plot(iterations[1*notFirst:], cmX[1*notFirst:], '--bo' )
  plt.title("CM_x")
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
import numpy as np
import sys, time, os, inspect, datetime
#import h5py as h5
import matplotlib.pyplot as plt

def plotData( maxVals, sumConc ):
  plt. figure(0)
  plt.clf()
  plt.plot(sumConc)
  
  plt.figure(1)
  plt.clf()
  plt.yscale("log")
  plt.plot(maxVals, "b")
  plt.draw()
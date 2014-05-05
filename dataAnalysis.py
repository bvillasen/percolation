import numpy as np
import sys, time, os, inspect, datetime
#import h5py as h5
import matplotlib.pyplot as plt

def plotData( maxVals, sumConc, iterations ):
  plt. figure(0)
  plt.clf()
  plt.plot(iterations, sumConc )
  plt.title(r"Concentration Sum")
  ax = plt.gca()
  ax.set_xlabel(r"Time")
  
  plt.figure(1)
  plt.clf()
  plt.yscale("log")
  plt.plot(iterations, maxVals, "b")
  plt.title(r"Concentration Max Value")
  ax = plt.gca()
  ax.set_xlabel(r"Time")
  plt.draw()
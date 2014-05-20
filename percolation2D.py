import sys, time, os
import numpy as np
import pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import h5py as h5
#import pycuda.curandom as curandom

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
animation2DDirectory = parentDirectory + "/animation2D"
sys.path.extend( [toolsDirectory, animation2DDirectory] )
from tools import printProgressTime, ensureDirectory
from cudaTools import setCudaDevice, getFreeMemory, kernelMemoryInfo, gpuArray2DtocudaArray
from dataAnalysis import plotConc, plotCM

nPoints = 512
probability = 0.35
hx = 0.5

cudaP = "double"
devN = None
usingAnimation = False
showKernelMemInfo = False
plottingConc = False
plottingCM = False

#Read in-line parameters
for option in sys.argv:
  if option.find("dev=") != -1: devN = int(option[-1]) 
  if option == "double": cudaP = "double"
  if option == "float": cudaP = "float"
  if option.find("mem") >=0: showKernelMemInfo = True
  if option.find("anim") >=0: usingAnimation = True
  if option.find("plotConc") >=0: plottingConc = True
  if option.find("plotCM") >=0: plottingCM = True
  if option.find("p=") >=0: probability = float(option[option.find("=")+1:])
  if option.find("h=") >=0: hx = float(option[option.find("=")+1:])
precision  = {"float":np.float32, "double":np.float64} 
cudaPre = precision[cudaP]
  
#set simulation dimentions 
nWidth = nPoints *2
nHeight = nPoints *2
#Lx, Ly = 1.,  1.
dx, dy = 1., 1.
xMin, yMin = 0., 0.

nCenter = 1
offsetX = -nWidth/2 + 128
offsetY = 0

iterationsPerPlot = 500
maxVals = []
sumConc = []
boundarySum = []
iterations = []
iterationsConc = []
cmX_list = []
cmY_list = []
showCM = plottingCM

#Initialize openGL
if usingAnimation:
  import animation2D
  animation2D.nWidth = nWidth
  animation2D.nHeight = nHeight
  animation2D.windowTitle = "Percolation 2D  grid={0}x{1}   p={2:.3f}     h={3:.3} ".format(nHeight, nWidth, float(probability), float(hx) )
  animation2D.initGL()
#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=devN, usingAnimation=usingAnimation )

#set thread grid for CUDA kernels
block_size_x, block_size_y  = 16, 16   
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )  
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
block2D = (block_size_x, block_size_y, 1)
grid2D = (gridx, gridy, 1)

#Read and compile CUDA code
print "\nCompiling CUDA code\n"
cudaCodeString_raw = open("cudaPercolation2D.cu", "r").read().replace("cudaP", cudaP)
cudaCodeString = cudaCodeString_raw  % { "THREADS_PER_BLOCK":block2D[0]*block2D[1], "B_WIDTH":block2D[0], "B_HEIGHT":block2D[1] }
cudaCode = SourceModule(cudaCodeString)
mainKernel_tex = cudaCode.get_function("main_kernel_tex" )
mainKernel_sh = cudaCode.get_function("main_kernel_shared" )
getCM_step1Kernel = cudaCode.get_function("getCM_step1_kernel")
tex_isFree = cudaCode.get_texref('tex_isFree')
tex_concentrationIn = cudaCode.get_texref('tex_concentrationIn')
if showKernelMemInfo: 
  kernelMemoryInfo(mainKernel_tex, 'mainKernel_tex')
  print ""
  kernelMemoryInfo(mainKernel_sh, 'mainKernel_shared')
  print ""
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
scalePlotData = ElementwiseKernel(arguments="cudaP a,  cudaP *realArray".replace("cudaP", cudaP),
				operation = "realArray[i] = log10( 1 + (a*realArray[i] ) ) ",
				name = "multiplyByScalarReal_kernel")
###########################################################################
def startTerrain( p ):
  global randomVals_h, isFree_h, concentration_h, isFree_h, isFree_d
  global concentrationIn_d, concentrationOut_d, blockBoundarySum_d
  randomVals_h = np.random.random([nHeight, nWidth])
  isFree_h = ( randomVals_h > p )
  concentration_h = np.zeros( [nHeight, nWidth], dtype=cudaPre )
  if nCenter==1:
    isFree_h[ offsetY + nHeight/2 - 1, offsetX + nWidth/2 - 1 ] = np.uint8(1)
    concentration_h[ offsetY + nHeight/2 - 1, offsetX + nWidth/2 - 1] = 1.
  else:
    isFree_h[ offsetY + nHeight/2 - nCenter/2 : offsetY + nHeight/2 + nCenter/2,
	      offsetX + nWidth/2  - nCenter/2 : offsetX + nWidth/2  +nCenter/2 ] = np.uint8(1)
    concentration_h[ offsetY + nHeight/2 - nCenter/2 : offsetY + nHeight/2 + nCenter/2,
		    offsetX + nWidth/2  - nCenter/2 : offsetX + nWidth/2  + nCenter/2 ] = 1./nCenter**2
  if cudaP == "double": isFree_d.set( isFree_h.astype(np.uint8) ) 
  if cudaP == "float": isFree_d.set( sFree_h.astype(np.int32) )
  concentrationIn_d.set( concentration_h )
  concentrationOut_d.set( concentration_h )
  blockBoundarySum_d.set( np.zeros( [ grid2D[1], grid2D[0] ], dtype=np.float32))
###########################################################################
nIter = 0
def oneIteration_tex():
  global nIter
  mainKernel_tex( np.int32(nWidth), np.int32(nHeight), cudaPre(hx), isFree_d, concentrationOut_d, 
		 grid=grid2D, block=block2D, texrefs=[tex_isFree, tex_concentrationIn] )
  copy2D_concentrationArray1( aligned=True )
  nIter += 1
def oneIteration_sh():
  global nIter
  mainKernel_sh( np.int32(nWidth), np.int32(nHeight), cudaPre(hx), isFree_d, 
		concentrationIn_d, concentrationOut_d, blockBoundarySum_d, grid=grid2D, block=block2D )
  mainKernel_sh( np.int32(nWidth), np.int32(nHeight), cudaPre(hx), isFree_d,
		concentrationOut_d, concentrationIn_d, blockBoundarySum_d, grid=grid2D, block=block2D )
  nIter += 1
###########################################################################
def getCM():
  getCM_step1Kernel( cudaPre(xMin), cudaPre(yMin), cudaPre(dx), cudaPre(dy), 
		    concentrationIn_d, cmX_d, cmY_d, grid=grid2D, block=block2D )
  return ( gpuarray.sum(cmX_d).get(), gpuarray.sum(cmY_d).get() )
def saveCM():
  if usingAnimation:
    animation2D.onePoint = getCM()
    if animIter%5==0:
      iterations.append( animIter*iterationsPerPlot )
      cmX_list.append( animation2D.onePoint[0] )
      cmY_list.append( animation2D.onePoint[1] )
      if plottingCM: plotCM( cmX_list, cmY_list, iterations, probability, hx, plotY=False, notFirst=True, sqrtX=False )
def saveConc( maxVal ):
    maxVals.append( maxVal )
    sumConc.append( gpuarray.sum(concentrationOut_d).get() )
    iterationsConc.append( iterationsPerPlot*animIter )
    boundarySum.append( gpuarray.sum(blockBoundarySum_d).get() )
    plotConc( maxVals, sumConc, boundarySum,  iterationsConc )
###########################################################################
###########################################################################
#For animation
animIter = 0
def stepFunction():
  global animIter
  cuda.memcpy_dtod( plotData_d.ptr, concentrationOut_d.ptr, concentrationOut_d.nbytes )
  maxVal = gpuarray.max( plotData_d ).get()
  scalePlotData(1e10/maxVal, plotData_d)
  if showCM: saveCM()
  if plottingConc and animIter%25 == 0: saveConc( maxVal )
  if cudaP == "float":  [ oneIteration_tex() for i in range(iterationsPerPlot) ]
  if cudaP == "double": [ oneIteration_sh() for i in range(iterationsPerPlot//2) ]  
  animIter += 1
  #print animIter*iterationsPerPlot 
def keyboardFunc(*args):
  global showCM
  ESCAPE = '\033'
  if args[0] == ESCAPE:
    print "Ending Simulation"
    sys.exit()   
  if args[0] == "c":
    showCM = not showCM
    animation2D.showPoint = not animation2D.showPoint
  if args[0] == "i": print "Iterations: {0}".format(animIter*iterationsPerPlot)
def specialKeyboardFunc( key, x, y ):
  global hx
  if key== animation2D.GLUT_KEY_LEFT:
    hx -= 0.005
  if key== animation2D.GLUT_KEY_RIGHT:
    hx += 0.005
  animation2D.windowTitle = "Percolation 2D  grid={0}x{1}   p={2:.3f}     h={3:.3} ".format(nHeight, nWidth, float(probability), float(hx) )
###########################################################################
###########################################################################
#Initialize Data
nData = nWidth*nHeight
print "Initializing CUDA memory"
np.random.seed(int(time.time()))  #Change numpy random seed
initialFreeMemory = getFreeMemory( show=True )
randomVals_h = np.random.random([nHeight, nWidth])
isFree_h = ( randomVals_h > probability )
concentration_h = np.zeros( [nHeight, nWidth], dtype=cudaPre )
if cudaP == "double": isFree_d = gpuarray.to_gpu( np.zeros( [nHeight, nWidth] , dtype=np.uint8 ) ) 
if cudaP == "float": isFree_d = gpuarray.to_gpu( np.zeros( [nHeight, nWidth] , dtype=np.int32 ) )
concentrationIn_d = gpuarray.to_gpu( concentration_h )
concentrationOut_d = gpuarray.to_gpu( concentration_h )
blockBoundarySum_d = gpuarray.to_gpu( np.zeros( [ grid2D[1], grid2D[0] ], dtype=np.float32)) 
startTerrain( probability ) 
#For CM calculation
cmX_d = gpuarray.to_gpu( concentration_h )
cmY_d = gpuarray.to_gpu( concentration_h )
#For texture version
if cudaP == "float":
  isFree_dArray, copy2D_isFreeArray   = gpuArray2DtocudaArray( isFree_d )
  tex_isFree.set_array( isFree_dArray )
  concentration1_dArray, copy2D_concentrationArray1 = gpuArray2DtocudaArray( concentrationOut_d )
  tex_concentrationIn.set_array( concentration1_dArray )
#For animation
if usingAnimation: plotData_d = gpuarray.to_gpu( np.zeros_like(concentration_h) )
finalFreeMemory = getFreeMemory( show=False )
print  " Total global memory used: {0:0.0f} MB\n".format( float(initialFreeMemory - finalFreeMemory)/1e6 ) 
###########################################################################
###########################################################################
#configure animation2D functions and plotData
if usingAnimation:
  animation2D.stepFunc = stepFunction
  animation2D.specialKeys = specialKeyboardFunc
  animation2D.keyboard = keyboardFunc
  if cudaP == "double": animation2D.usingDouble = True
  if showCM: animation2D.showPoint = True
  animation2D.plotData_d = plotData_d
  animation2D.background_h = isFree_h
  animation2D.backgroundType = "move"
  animation2D.maxVar = cudaPre(10.00001)
  animation2D.minVar = cudaPre(0.)
###########################################################################
###########################################################################
if showKernelMemInfo: 
  if cudaP == "float": oneIteration_tex() 
  else: oneIteration_sh()
  print "Precision: ", cudaP
  print "Timing Info saved in: cuda_profile_1.log \n\n"
  sys.exit()
###########################################################################
###########################################################################
#Start Simulation
if plottingConc or plottingCM: plt.ion(), plt.show(), 
print "Starting simulation"
if cudaP == "double": print "Using double precision"
else: print "Using single precision"
print "Grid: {0} x {1}\n".format(nHeight, nWidth)


##run animation
if usingAnimation:
  print "p = {0:1.2f}".format( probability ) 
  print "h = {0:1.2f}\n".format( hx ) 
  animation2D.animate()

dataDir = currentDirectory + "/data/"
ensureDirectory( dataDir )

probability_job = [ 0.20, 0.28 ]
if devN == 1: hx_job = [ 0.25 ]
#hx_job = [ 0.25, 0.26, 0.27, 0.28, 0.29 ]
#hx_job = [  0.28, 0.29 ]
#hx_job = [  0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99 ]
#hx_job = [  0.52, 0.54, 0.56, 0.58, 0.60, 0.63, 0.66, 0.69, 0.72 ] 
#hx_job = [ 0.40, 0.41, 0.42, 0.43, 0.44, 0.45 ]
#hx_job = [ 0.46, 0.48, 0.50 ]
#hx_job = [ 0.47, 0.50, 0.55, 0.60, 0.65, 0.70 ]
#hx_job = [ 0.75, 0.80, 0.85, 0.9, 0.95, 0.99 ]
#hx_job = [ 0.50, 0.39, 0.41, 0.43, 0.45, 0.47, 0.49, 0.52, 0.54 ] 
#hx_job = [  0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95 ]
print "p = {0}".format( probability_job ) 
print "h = {0}\n".format( hx_job ) 

nRealizations = 100
nRuns = 50
iterationsPerRun = 400
iterations = np.arange(nRuns)*iterationsPerRun
print "nRealizations: {2}\n nRuns: {0}\n  Iterations per Run: {1}\n Time: {3}\n".format( nRuns, iterationsPerRun, nRealizations, nRuns*iterationsPerRun ) 

dataFiles = []

start, end = cuda.Event(), cuda.Event()
start.record()
counter = 0
for probability in probability_job:
  for hx in hx_job:
    boundarySum_all = []
    CM_all = []
    for iterNumber in range(nRealizations):
      CM_list = []
      startTerrain( probability )
      printProgressTime( counter*nRealizations + iterNumber, len(probability_job)*len(hx_job)*nRealizations, start.time_till(end.record().synchronize())*1e-3 )
      for runNumber in range(nRuns):
	CM_list.append(getCM())
	if cudaP == "float":  [ oneIteration_tex() for i in range(iterationsPerRun) ]
	if cudaP == "double": [ oneIteration_sh()  for i in range(iterationsPerRun//2) ]
      CM_dataFromRun = np.array(CM_list).T
      CM_all.append(CM_dataFromRun)
      boundarySum_all.append( gpuarray.sum(blockBoundarySum_d).get() )
    counter += 1
    #Save data
    CM_data = np.array( CM_all )
    boundary_data = np.array( boundarySum_all ) 
    dataFileName = dataDir + "p_{0:.0f}h_{1:.0f}H_{2}W_{3}R_{4}.hdf5".format( float(probability*100), float(hx*100), nHeight, nWidth, nRealizations )
    dataFiles.append(dataFileName[dataFileName.find("data/"):])
    dataFile = h5.File(dataFileName,'w')
    dataFile.create_dataset( "iterations", data=iterations, compression='lzf')
    dataFile.create_dataset( "CM_data", data=CM_data, compression='lzf')
    dataFile.create_dataset( "boundary_data", data=boundary_data, compression='lzf')
    dataFile.close()

print "\n\nFinished in : {0:.4f}  sec\n".format( float( start.time_till(end.record().synchronize())*1e-3 ) ) 

print "Data Saved:"
for fileName in dataFiles:
  print " ", fileName 
print ""

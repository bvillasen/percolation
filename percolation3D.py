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
volumeRenderDirectory = parentDirectory + "/volumeRender"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )
from tools import printProgressTime, ensureDirectory
from cudaTools import setCudaDevice, getFreeMemory, kernelMemoryInfo, gpuArray3DtocudaArray
from dataAnalysis import plotConc, plotCM

nPoints = 128
probability = 0.62
hx = 0.4

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
  if option.find("mem") >=0: showKernelMemInfo = True
  if option.find("anim") >=0: usingAnimation = True
  if option.find("plotConc") >=0: plottingConc = True
  if option.find("plotCM") >=0: plottingCM = True
  if option.find("p=") >=0: probability = float(option[option.find("=")+1:])
  if option.find("h=") >=0: hx = float(option[option.find("=")+1:])
precision  = {"float":np.float32, "double":np.float64} 
cudaPre = precision[cudaP]

#set simulation dimentions 
nWidth = nPoints * 2
nHeight = nPoints
nDepth = nPoints
#Lx, Ly = 1.,  1.
dx, dy, dz = 1., 1., 1.
xMin, yMin, zMin = 0., 0., 0.

nCenter = 1
offsetX = -nWidth/2 + 128
offsetY = 0
offsetZ = 0

iterationsPerPlot = 20
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
  import volumeRender
  volumeRender.nWidth = nWidth
  volumeRender.nHeight = nHeight
  volumeRender.nDepth = nDepth
  volumeRender.scaleX = nWidth/nPoints
  volumeRender.windowTitle = "Percolation 3D  grid={0}x{1}X{2}   p={3:.3f}     h={4:.3} ".format( nDepth, nHeight, nWidth,  float(probability), float(hx) )
  volumeRender.initGL()
#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=devN, usingAnimation=usingAnimation )

#set thread grid for CUDA kernels
block_size_x, block_size_y, block_size_z = 8,8,8   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
gridz = nDepth // block_size_z + 1 * ( nDepth % block_size_z != 0 )
block3D = (block_size_x, block_size_y, block_size_z)
grid3D = (gridx, gridy, gridz)

#Read and compile CUDA code
print "\nCompiling CUDA code\n"
cudaCodeString_raw = open("cudaPercolation3D.cu", "r").read().replace("cudaP", cudaP)
cudaCodeString = cudaCodeString_raw  % { "THREADS_PER_BLOCK":block3D[0]*block3D[1]*block3D[2], "B_WIDTH":block3D[0], "B_HEIGHT":block3D[1], "B_DEPTH":block3D[2] }
cudaCode = SourceModule(cudaCodeString)
mainKernel_sh = cudaCode.get_function("main_kernel_shared" )
getCM_step1Kernel = cudaCode.get_function("getCM_step1_kernel")
if showKernelMemInfo: 
  kernelMemoryInfo(mainKernel_sh, 'mainKernel_shared')
  print ""
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
scalePlotData = ElementwiseKernel(arguments="cudaP a,  cudaP *dataIn, cudaP *dataOut".replace("cudaP", cudaP),
				operation = "dataOut[i] = log10( 1 + (a*dataIn[i] ) ) ",
				name = "scalePlotData_kernel")
########################################################################
floatToUchar = ElementwiseKernel(arguments="cudaP *input, unsigned char *output".replace("cudaP", cudaP),
				operation = "output[i] = (unsigned char) ( -255*(0.088*input[i]-1));",
				name = "floatToUchar_kernel")
########################################################################
def startTerrain( p ):
  global randomVals_h, isFree_h, concentration_h, isFree_h, isFree_d
  global concentrationIn_d, concentrationOut_d, blockBoundarySum_d
  randomVals_h = np.random.random([nDepth, nHeight, nWidth])
  isFree_h = ( randomVals_h > p )
  concentration_h = np.zeros( [nDepth, nHeight, nWidth], dtype=cudaPre )
  if nCenter==1:
    isFree_h[ offsetZ + nDepth/2 - 1, offsetY + nHeight/2 - 1, offsetX + nWidth/2 - 1 ] = np.uint8(1)
    concentration_h[ offsetZ + nDepth/2 - 1, offsetY + nHeight/2 - 1, offsetX + nWidth/2 - 1] = 1.
  else:
    isFree_h[ offsetZ + nDepth/2  - nCenter/2 : offsetZ + nDepth/2  + nCenter/2,
	      offsetY + nHeight/2 - nCenter/2 : offsetY + nHeight/2 + nCenter/2,
	      offsetX + nWidth/2  - nCenter/2 : offsetX + nWidth/2  +nCenter/2 ] = np.uint8(1)
    concentration_h[offsetZ + nDepth/2  - nCenter/2 : offsetZ + nDepth/2  + nCenter/2, 
                    offsetY + nHeight/2 - nCenter/2 : offsetY + nHeight/2 + nCenter/2,
		    offsetX + nWidth/2  - nCenter/2 : offsetX + nWidth/2  + nCenter/2 ] = 1./nCenter**3
  isFree_d.set( isFree_h.astype(np.uint8) ) 
  concentrationIn_d.set( concentration_h )
  concentrationOut_d.set( concentration_h )
  blockBoundarySum_d.set( np.zeros( [ grid3D[2], grid3D[1], grid3D[0] ], dtype=np.float32))
###########################################################################
nIter = 0
def oneIteration_sh():
  global nIter
  mainKernel_sh( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), cudaPre(hx), isFree_d, 
		concentrationIn_d, concentrationOut_d, blockBoundarySum_d, grid=grid3D, block=block3D )
  mainKernel_sh( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), cudaPre(hx), isFree_d,
		concentrationOut_d, concentrationIn_d, blockBoundarySum_d, grid=grid3D, block=block3D )
  nIter += 1
###########################################################################
def getCM():
  getCM_step1Kernel( cudaPre(xMin), cudaPre(yMin), cudaPre(zMin), cudaPre(dx), cudaPre(dy), cudaPre(dz), 
		    concentrationIn_d, cmX_d, cmY_d, cmZ_d, grid=grid3D, block=block3D )
  return ( gpuarray.sum(cmX_d).get(), gpuarray.sum(cmY_d).get(), gpuarray.sum(cmZ_d).get() )
def saveCM():
  if usingAnimation:
    #animation2D.onePoint = getCM()
    if animIter%5==0:
      iterations.append( animIter*iterationsPerPlot )
      cmX_list.append( animation2D.onePoint[0] )
      cmY_list.append( animation2D.onePoint[1] )
###########################################################################
###########################################################################
def sendToScreen():
  maxVal = gpuarray.max( concentrationIn_d ).get()
  scalePlotData( 1e10/maxVal, concentrationIn_d, plotData_d )
  floatToUchar( plotData_d, plotDataChar_d)  #There is a 0.1 scaling factor in this operation
  copyToScreenArray()
########################################################################
#For animation
animIter = 0
def stepFunction():
  global animIter
  sendToScreen()
  #if showCM: saveCM()
  #if plottingConc and animIter%25 == 0: saveConc( maxVal )
  [ oneIteration_sh() for i in range(iterationsPerPlot//2) ]  
  animIter += 1
###########################################################################
###########################################################################
#Initialize Data
nData = nWidth*nHeight
print "Initializing CUDA memory"
np.random.seed(int(time.time()))  #Change numpy random seed
initialFreeMemory = getFreeMemory( show=True )
randomVals_h = np.random.random([nDepth, nHeight, nWidth])
isFree_h = ( randomVals_h > probability )
concentration_h = np.zeros( [nDepth, nHeight, nWidth], dtype=cudaPre )
isFree_d = gpuarray.to_gpu( np.zeros( [nDepth, nHeight, nWidth] , dtype=np.uint8 ) ) 
concentrationIn_d = gpuarray.to_gpu( concentration_h )
concentrationOut_d = gpuarray.to_gpu( concentration_h )
blockBoundarySum_d = gpuarray.to_gpu( np.zeros( [ grid3D[2], grid3D[1], grid3D[0] ], dtype=np.float32)) 
startTerrain( probability ) 
#For CM calculation
cmX_d = gpuarray.to_gpu( concentration_h )
cmY_d = gpuarray.to_gpu( concentration_h )
cmZ_d = gpuarray.to_gpu( concentration_h )
#For animation
if usingAnimation: 
  plotData_d = gpuarray.to_gpu( np.zeros([nDepth, nHeight, nWidth], dtype = cudaPre) )
  plotDataChar_d = gpuarray.to_gpu( np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8) )
  volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotDataChar_d )
finalFreeMemory = getFreeMemory( show=False )
print  " Total global memory used: {0:0.0f} MB\n".format( float(initialFreeMemory - finalFreeMemory)/1e6 ) 
###########################################################################
###########################################################################
#configure volumeRender functions 
if usingAnimation:
  volumeRender.stepFunc = stepFunction
  #volumeRender.specialKeys = specialKeyboardFunc
###########################################################################
###########################################################################
if showKernelMemInfo: 
  oneIteration_sh()
  print "Precision: ", cudaP
  print "Timing Info saved in: cuda_profile_1.log \n\n"
  sys.exit()
###########################################################################
###########################################################################
#Start Simulation
if plottingConc or plottingCM: plt.ion(), plt.show(), 
print "Starting simulation"
print "Using double precision"
print "Grid: {0} x {1} x {2} \n".format(nDepth, nHeight, nWidth)

##run animation
if usingAnimation:
  print "p = {0:1.2f}".format( probability ) 
  print "h = {0:1.2f}\n".format( hx ) 
  volumeRender.animate()

dataDir = currentDirectory + "/data3D/"
ensureDirectory( dataDir )
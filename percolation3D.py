import sys, time, os
import numpy as np
import pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
#import pycuda.curandom as curandom

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
volumeRenderDirectory = parentDirectory + "/volumeRender"
sys.path.extend( [toolsDirectory, volumeRenderDirectory] )
import volumeRender
from cudaTools import setCudaDevice, getFreeMemory, kernelMemoryInfo, gpuArray3DtocudaArray
from dataAnalysis import plotData

nPoints = 128
probability = 0.04

cudaP = "float"
devN = None
usingAnimation = True
showKernelMemInfo = False
plotting = False

#Read in-line parameters
for option in sys.argv:
  if option.find("device=") != -1: devN = int(option[-1]) 
  if option == "double": cudaP = "double"
  if option == "float": cudaP = "float"
  if option.find("mem") >=0: showKernelMemInfo = True
  if option.find("anim") >=0: usingAnimation = True
  if option.find("plot") >=0: plotting = True
  if option.find("p=") >=0: probability = float(option[option.find("=")+1:])

precision  = {"float":np.float32, "double":np.float64} 
cudaPre = precision[cudaP]
  
#set simulation dimentions 
nWidth = nPoints
nHeight = nPoints 
nDepth = nPoints

nCenter = 16  #Has to be even
offsetX = 0
offsetY = 0
offsetZ = 0

nIterationsPerPlot = 100
maxVals = []
sumConc = []

#Initialize openGL
if usingAnimation:
  volumeRender.nWidth = nWidth
  volumeRender.nHeight = nHeight
  volumeRender.nDepth = nDepth
  volumeRender.windowTitle = "Percolation 3D  points={0}x{1}x{2}   p={3:.2f}".format(nHeight, nWidth, nDepth, float(probability))
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
#countFreeNeighborsKernel = cudaCode.get_function("countFreeNeighbors_kernel")
mainKernel_tex = cudaCode.get_function("main_kernel_tex" )
#mainKernel_sh = cudaCode.get_function("main_kernel_shared" )
tex_isFree = cudaCode.get_texref('tex_isFree')
#tex_nNeighb = cudaCode.get_texref('tex_nNeighb')
tex_concentrationIn = cudaCode.get_texref('tex_concentrationIn')
##surf_concentrationOut = cudaCode.get_surfref('surf_concentrationOut')
if showKernelMemInfo: 
  kernelMemoryInfo(mainKernel_tex, 'mainKernel_tex')
  print ""
  #kernelMemoryInfo(mainKernel_sh, 'mainKernel_shared')
  #print ""
  #sys.exit()
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
multiplyByScalarReal = ElementwiseKernel(arguments="cudaP a, cudaP *realArray".replace("cudaP", cudaP),
				operation = "realArray[i] = a*realArray[i] ",
				name = "multiplyByScalarReal_kernel")
###########################################################################
floatToUchar = ElementwiseKernel(arguments="float *input, unsigned char *output",
				operation = "output[i] = (unsigned char) ( -255*(input[i]-1));",
				name = "floatToUchar_kernel")
########################################################################
def sendToScreen( plotData ):
  floatToUchar( plotDataFloat_d, plotData_d )
  copyToScreenArray()
###########################################################################
nIter = 0
def oneIteration_tex():
  global nIter
  mainKernel_tex( np.int32(nWidth), np.int32(nHeight), np.int32(nDepth), isFree_d, concentrationOut_d, 
	          grid=grid3D, block=block3D, texrefs=[tex_isFree, tex_concentrationIn] )
  copy3D_concentrationArray1()
  nIter += 1
###########################################################################
animIter = 0
def stepFunction():
  global animIter
  cuda.memcpy_dtod( plotDataFloat_d.ptr, concentrationOut_d.ptr, concentrationOut_d.nbytes )
  maxVal = (gpuarray.max(plotDataFloat_d)).get()
  multiplyByScalarReal( cudaPre(0.5/(maxVal)), plotDataFloat_d )
  floatToUchar( plotDataFloat_d, plotDataChars_d)
  copyToScreenArray()
  if cudaP == "float": [ oneIteration_tex() for i in range(nIterationsPerPlot) ]
  #else: [ oneIteration_sh() for i in range(nIterationsPerPlot//2) ]
  if plotting and animIter%25 == 0: 
    maxVals.append( maxVal )
    sumConc.append( gpuarray.sum(concentrationIn_d).get() )
    plotData( maxVals, sumConc )
  animIter += 1
###########################################################################
###########################################################################
#Initialize Data
nData = nWidth*nHeight*nDepth
print "Initializing CUDA memory"
np.random.seed(int(time.time()))  #Change numpy random seed
initialFreeMemory = getFreeMemory( show=True )
randomVals_h = np.random.random([nDepth, nHeight, nWidth])
isFree_h = ( randomVals_h > probability )
concentration_h = np.zeros( [nDepth, nHeight, nWidth], dtype=cudaPre )
if nCenter==1:
  isFree_h[offsetZ + nDepth/2 - 1, offsetY + nHeight/2 - 1, offsetX + nWidth/2 - 1 ] = True
  concentration_h[offsetZ + nDepth/2 -1, offsetY + nHeight/2 - 1, offsetX +nWidth/2 - 1 ] = 1.
else:
  isFree_h[ offsetZ + nDepth/2  - nCenter/2 : offsetZ + nDepth/2  + nCenter/2,
	    offsetY + nHeight/2 - nCenter/2 : offsetY + nHeight/2 + nCenter/2,
	    offsetX + nWidth/2  - nCenter/2 : offsetX + nWidth/2  +nCenter/2 ] = True
  concentration_h[ offsetZ + nDepth/2  - nCenter/2 : offsetZ + nDepth/2  + nCenter/2,
		   offsetY + nHeight/2 - nCenter/2 : offsetY + nHeight/2 + nCenter/2,
		   offsetX + nWidth/2  - nCenter/2 : offsetX + nWidth/2  + nCenter/2 ] = 1./nCenter**3
isFree_d = gpuarray.to_gpu( isFree_h.astype(np.int32) ) 
concentrationIn_d = gpuarray.to_gpu( concentration_h )
concentrationOut_d = gpuarray.to_gpu( concentration_h )
#For texture version
isFree_dArray, copy3D_isFreeArray   = gpuArray3DtocudaArray( isFree_d )
tex_isFree.set_array( isFree_dArray )
if cudaP == "float":
  concentration1_dArray, copy3D_concentrationArray1 = gpuArray3DtocudaArray( concentrationOut_d, allowSurfaceBind=True )
  tex_concentrationIn.set_array( concentration1_dArray )
#memory for plotting
plotDataFloat_d = gpuarray.to_gpu(np.zeros_like(concentration_h))
plotDataChars_d = gpuarray.to_gpu(np.zeros([nDepth, nHeight, nWidth], dtype = np.uint8))
volumeRender.plotData_dArray, copyToScreenArray = gpuArray3DtocudaArray( plotDataChars_d )
finalFreeMemory = getFreeMemory( show=False )
print  " Total global memory used: {0:0.0f} MB\n".format( float(initialFreeMemory - finalFreeMemory)/1e6 ) 
###########################################################################
###########################################################################
#configure volumeRender functions 
if usingAnimation:
  volumeRender.stepFunc = stepFunction
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
if plotting: plt.ion(), plt.show(), 
print "Starting simulation"
if cudaP == "double": print "Using double precision\n"
else: print "Using single precision\n"
print "p = {0:1.2f}\n".format( probability ) 

#countFreeNeighbors()

#run volumeRender animation
if usingAnimation:
  volumeRender.animate()
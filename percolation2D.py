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
animation2DDirectory = parentDirectory + "/animation2D"
sys.path.extend( [toolsDirectory, animation2DDirectory] )
import animation2D
from cudaTools import setCudaDevice, getFreeMemory, kernelMemoryInfo, gpuArray2DtocudaArray

nPoints = 512
probability = .4

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

precision  = {"float":np.float32, "double":np.float64} 
cudaPre = precision[cudaP]
  
#set simulation dimentions 
nWidth = nPoints
nHeight = nPoints 

nCenter = 64
offsetX = 0
offsetY = 0

nIterationsPerPlot = 400
showActivity = np.uint8(0)

#Initialize openGL
if usingAnimation:
  animation2D.nWidth = nWidth
  animation2D.nHeight = nHeight
  animation2D.windowTitle = "Percolation 2D  points={0}x{1}   p={2:.2f}".format(nHeight, nWidth, float(probability))
  animation2D.initGL()
#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=devN, usingAnimation=usingAnimation )

#set thread grid for CUDA kernels
block_size_x, block_size_y  = 16, 16   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )  
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
block2D = (block_size_x, block_size_y, 1)
grid2D = (gridx, gridy, 1)

#Read and compile CUDA code
print "\nCompiling CUDA code\n"
cudaCodeString_raw = open("cudaPercolation2D.cu", "r").read().replace("cudaP", cudaP)
cudaCodeString = cudaCodeString_raw  % { "THREADS_PER_BLOCK":block2D[0]*block2D[1], "B_WIDTH":block2D[0], "B_HEIGHT":block2D[1] }
cudaCode = SourceModule(cudaCodeString)
countFreeNeighborsKernel = cudaCode.get_function("countFreeNeighbors_kernel")
mainKernel_tex = cudaCode.get_function("main_kernel_tex" )
mainKernel_sh = cudaCode.get_function("main_kernel_shared" )
findActivityKernel = cudaCode.get_function( "findActivity_kernel" )
getActivityKernel = cudaCode.get_function( "getActivity_kernel" )
tex_isFree = cudaCode.get_texref('tex_isFree')
tex_nNeighb = cudaCode.get_texref('tex_nNeighb')
tex_concentration = cudaCode.get_texref('tex_concentration')
if showKernelMemInfo: 
  kernelMemoryInfo(mainKernel_tex, 'mainKernel_tex')
  print ""
  kernelMemoryInfo(mainKernel_sh, 'mainKernel_shared')
  sys.exit()
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
multiplyByScalarReal = ElementwiseKernel(arguments="cudaP a, cudaP *realArray".replace("cudaP", cudaP),
				operation = "realArray[i] = a*realArray[i] ",
				name = "multiplyByScalarReal_kernel")
###########################################################################
def countFreeNeighbors():
  tex_isFree.set_array( isFree_dArray )
  tex_nNeighb.set_array( nNeighb_dArray )	
  countFreeNeighborsKernel( np.int32(nWidth), np.int32(nHeight), nNeighb_d, grid=grid2D, block=block2D )
  copy2D_nNeigdbArray( aligned=True )
###########################################################################
nIter = 0
def oneIteration_tex():
  global nIter
  #findActivityKernel( cudaPre(1.e-30), concentrationOut_d, activeBlocks_d, grid=grid2D, block=block2D  )
  mainKernel_tex( np.int32(nWidth), np.int32(nHeight), isFree_d, concentrationOut_d, activeBlocks_d,
	         plotData_d, showActivity, grid=grid2D, block=block2D )
  copy2D_concentrationArray( aligned=True )
  nIter += 1
def oneIteration_sh():
  global nIter
  mainKernel_sh( np.int32(nWidth), np.int32(nHeight), isFree_d, concentrationIn_d, concentrationOut_d, nNeighb_d,  grid=grid2D, block=block2D )
  mainKernel_sh( np.int32(nWidth), np.int32(nHeight), isFree_d, concentrationOut_d, concentrationIn_d, nNeighb_d,  grid=grid2D, block=block2D )
  #cuda.memcpy_dtod( concentrationIn_d.ptr, concentrationOut_d.ptr, concentrationOut_d.nbytes )
  #concentrationIn_d.gpudata, concentrationOut_d.gpudata = concentrationOut_d.gpudata, concentrationIn_d.gpudata 
  nIter += 1
###########################################################################
animIter = 0
def stepFunction():
  global animIter
  if cudaP == "float": [ oneIteration_tex() for i in range(nIterationsPerPlot) ]
  else: [ oneIteration_sh() for i in range(nIterationsPerPlot//2) ]
  cuda.memcpy_dtod( plotData_d.ptr, concentrationOut_d.ptr, concentrationOut_d.nbytes )
  maxVal = gpuarray.max( plotData_d ).get()
  multiplyByScalarReal(1./maxVal, plotData_d)
  animIter += 1
###########################################################################
###########################################################################
#Initialize Data
nData = nWidth*nHeight
print "Initializing CUDA memory"
np.random.seed(int(time.time()))  #Change numpy random seed
initialFreeMemory = getFreeMemory( show=True )
randomVals_h = np.random.random([nHeight, nWidth])
isFree_h = ( randomVals_h > probability )
isFree_h[offsetY+nHeight/2-nCenter/2:offsetY+nHeight/2+nCenter/2,offsetX+nWidth/2-nCenter/2:offsetX+nWidth/2+nCenter/2 ] = True
isFree_d = gpuarray.to_gpu( isFree_h.astype(np.uint8) ) 
nNeighb_d = gpuarray.to_gpu( np.zeros([nHeight, nWidth], dtype=np.int32) )
concentration_h = np.zeros( [nHeight, nWidth], dtype=cudaPre )
concentration_h[offsetY+nHeight/2-nCenter/2:offsetY+nHeight/2+nCenter/2,offsetX+nWidth/2-nCenter/2:offsetX+nWidth/2+nCenter/2 ] = 1.
concentrationIn_d = gpuarray.to_gpu( concentration_h )
concentrationOut_d = gpuarray.to_gpu( concentration_h )
activeBlocks_d = gpuarray.to_gpu( np.zeros( [ grid2D[1],grid2D[0] ], dtype=np.uint8) )
activeThreads_d = gpuarray.to_gpu( np.zeros([nHeight, nWidth], dtype=np.uint8) )
#For texture version
isFree_dArray, copy2D_isFreeArray   = gpuArray2DtocudaArray( isFree_d )
nNeighb_dArray, copy2D_nNeigdbArray = gpuArray2DtocudaArray( nNeighb_d )
if cudaP == "float":
  concentration_dArray, copy2D_concentrationArray = gpuArray2DtocudaArray( concentrationOut_d )
  tex_concentration.set_array( concentration_dArray )
#For plotting
plotData_d = gpuarray.to_gpu( np.zeros_like(concentration_h) )
finalFreeMemory = getFreeMemory( show=False )
print  " Total global memory used: {0:0.0f} MB\n".format( float(initialFreeMemory - finalFreeMemory)/1e6 ) 
###########################################################################
###########################################################################
#configure animation2D functions and plotData
animation2D.stepFunc = stepFunction
#animation2D.specialKeys = specialKeyboardFunc
if cudaP == "double": animation2D.usingDouble = True
animation2D.plotData_d = plotData_d
animation2D.background_h = isFree_h
animation2D.maxVar = cudaPre(1.01)
animation2D.minVar = cudaPre(0.)
###########################################################################
###########################################################################
#Start Simulation
if plotting: plt.ion(), plt.show()
print "Starting simulation"
if cudaP == "double": print "Using double precision\n"
else: print "Using single precision\n"
print "p = {0:1.2f}\n".format( probability ) 
countFreeNeighbors()
#oneIteration()

#run animation
if usingAnimation:
  animation2D.animate()
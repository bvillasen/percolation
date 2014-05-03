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
from dataAnalysis import plotData

nPoints = 1024
probability = 0.
hx = 0.26

cudaP = "float"
devN = None
usingAnimation = False
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

nCenter = 1
offsetX = 0
offsetY = 0

nIterationsPerPlot = 400
maxVals = []
sumConc = []

#Initialize openGL
if usingAnimation:
  animation2D.nWidth = nWidth
  animation2D.nHeight = nHeight
  animation2D.windowTitle = "Percolation 2D  points={0}x{1}   p={2:.2f}".format(nHeight, nWidth, float(probability))
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
#findActivityKernel = cudaCode.get_function( "findActivity_kernel" )
#getActivityKernel = cudaCode.get_function( "getActivity_kernel" )
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
multiplyByScalarReal = ElementwiseKernel(arguments="cudaP a, cudaP *realArray".replace("cudaP", cudaP),
				operation = "realArray[i] = a*realArray[i] ",
				name = "multiplyByScalarReal_kernel")
###########################################################################
#def countFreeNeighbors():
  #tex_isFree.set_array( isFree_dArray )
  #tex_nNeighb.set_array( nNeighb_dArray )	
  #countFreeNeighborsKernel( np.int32(nWidth), np.int32(nHeight), nNeighb_d, grid=grid2D, block=block2D )
  #copy2D_nNeigdbArray( aligned=True )
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
  mainKernel_sh( np.int32(nWidth), np.int32(nHeight), cudaPre(hx), isFree_d, concentrationIn_d, concentrationOut_d,
		grid=grid2D, block=block2D, texrefs=[tex_isFree] )
  mainKernel_sh( np.int32(nWidth), np.int32(nHeight), cudaPre(hx), isFree_d, concentrationOut_d, concentrationIn_d,
		grid=grid2D, block=block2D, texrefs=[tex_isFree] )
  #cuda.memcpy_dtod( concentrationIn_d.ptr, concentrationOut_d.ptr, concentrationOut_d.nbytes )
  #concentrationIn_d.gpudata, concentrationOut_d.gpudata = concentrationOut_d.gpudata, concentrationIn_d.gpudata 
  nIter += 1
###########################################################################
animIter = 0
def stepFunction():
  global animIter
  cuda.memcpy_dtod( plotData_d.ptr, concentrationOut_d.ptr, concentrationOut_d.nbytes )
  maxVal = gpuarray.max( plotData_d ).get()
  multiplyByScalarReal(1./maxVal, plotData_d)
  if cudaP == "float": [ oneIteration_tex() for i in range(nIterationsPerPlot) ]
  else: [ oneIteration_sh() for i in range(nIterationsPerPlot//2) ]
  if plotting and animIter%25 == 0: 
    maxVals.append( maxVal )
    sumConc.append( gpuarray.sum(concentrationIn_d).get() )
    plotData( maxVals, sumConc )
  animIter += 1
  
def specialKeyboardFunc( key, x, y ):
  global hx
  if key== animation2D.GLUT_KEY_LEFT:
    hx -= 0.01
  if key== animation2D.GLUT_KEY_RIGHT:
    hx += 0.01
  animation2D.windowTitle = "Percolation 2D  grid={0}x{1}   p={2:.1f}     h={3:.2} ".format(nHeight, nWidth, float(probability), float(hx) )
  
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
if nCenter==1:
  isFree_h[ offsetY + nHeight/2 - 1, offsetX + nWidth/2 - 1 ] = np.uint8(1)
  concentration_h[ offsetY + nHeight/2 - 1, offsetX + nWidth/2 - 1] = 1.
else:
  isFree_h[ offsetY + nHeight/2 - nCenter/2 : offsetY + nHeight/2 + nCenter/2,
	    offsetX + nWidth/2  - nCenter/2 : offsetX + nWidth/2  +nCenter/2 ] = np.uint8(1)
  concentration_h[ offsetY + nHeight/2 - nCenter/2 : offsetY + nHeight/2 + nCenter/2,
		   offsetX + nWidth/2  - nCenter/2 : offsetX + nWidth/2  + nCenter/2 ] = 1./nCenter**2
isFree_d = gpuarray.to_gpu( isFree_h.astype(np.uint8) ) 
concentrationIn_d = gpuarray.to_gpu( concentration_h )
concentrationOut_d = gpuarray.to_gpu( concentration_h )
activeBlocks_d = gpuarray.to_gpu( np.zeros( [ grid2D[1],grid2D[0] ], dtype=np.uint8) )
activeThreads_d = gpuarray.to_gpu( np.zeros([nHeight, nWidth], dtype=np.uint8) )
#For texture version
isFree_dArray, copy2D_isFreeArray   = gpuArray2DtocudaArray( isFree_d )
tex_isFree.set_array( isFree_dArray )
if cudaP == "float":
  concentration1_dArray, copy2D_concentrationArray1 = gpuArray2DtocudaArray( concentrationOut_d )
  tex_concentrationIn.set_array( concentration1_dArray )
#For plotting
plotData_d = gpuarray.to_gpu( np.zeros_like(concentration_h) )
finalFreeMemory = getFreeMemory( show=False )
print  " Total global memory used: {0:0.0f} MB\n".format( float(initialFreeMemory - finalFreeMemory)/1e6 ) 
###########################################################################
###########################################################################
#configure animation2D functions and plotData
if usingAnimation:
  animation2D.stepFunc = stepFunction
  animation2D.specialKeys = specialKeyboardFunc
  if cudaP == "double": animation2D.usingDouble = True
  animation2D.plotData_d = plotData_d
  animation2D.background_h = isFree_h
  animation2D.backgroundType = "move"
  animation2D.maxVar = cudaPre(1.01)
  animation2D.minVar = cudaPre(0.)
  animation2D.windowTitle = "Percolation 2D  grid={0}x{1}   p={2:.2f}     h={3:.2} ".format(nHeight, nWidth, float(probability), float(hx) )

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
print "p = {0:1.2f}".format( probability ) 
print "h = {0:1.2f}\n".format( hx ) 

##run animation
if usingAnimation:
  animation2D.animate()
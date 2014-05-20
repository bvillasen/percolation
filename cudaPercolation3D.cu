#include <stdint.h>
#include <cuda.h>

typedef unsigned char uchar;

/////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////// 
__global__ void main_kernel_shared( const int nWidth, const int nHeight, const int nDepth, cudaP hx, uchar *isFreeAll,
			          cudaP *concIn, cudaP *concentrationOut, float *blockBoundarySum ){
  const int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  const int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  const int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  const int tid = t_j + t_i*blockDim.x*gridDim.x + t_k*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
  
  __shared__ float boundarySum;
  __shared__ uchar   isFree_sh[ %(B_WIDTH)s + 2 ][ %(B_HEIGHT)s + 2 ][ %(B_DEPTH)s + 2 ];
  __shared__ cudaP     conc_sh[ %(B_WIDTH)s + 2 ][ %(B_HEIGHT)s + 2 ][ %(B_DEPTH)s + 2 ];
  if (threadIdx.x == 0 and threadIdx.y == 0 and threadIdx.z == 0) boundarySum = blockBoundarySum[blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*blockDim.y*blockDim.x];
  conc_sh  [threadIdx.x+1][threadIdx.y+1][threadIdx.z+1] =    concIn[tid];
  isFree_sh[threadIdx.x+1][threadIdx.y+1][threadIdx.z+1] = isFreeAll[tid];
  __syncthreads();
  
  
  //LOAD NEIGHBORS
  //Left boundary
  if (t_j == 0){
    conc_sh  [0][threadIdx.y+1][threadIdx.z+1]   =    concIn[ (nWidth-1) + t_i*nWidth + t_k*nWidth*nHeight ];
    isFree_sh[0][threadIdx.y+1][threadIdx.z+1]   = isFreeAll[ (nWidth-1) + t_i*nWidth + t_k*nWidth*nHeight ];
    atomicAdd( &boundarySum , (float)conc_sh[1][threadIdx.y+1][threadIdx.z+1] );
  }
  else if ( threadIdx.x == 0 ){
    conc_sh  [0][threadIdx.y+1][threadIdx.z+1]   =    concIn[ (t_j-1) + t_i*nWidth + t_k*nWidth*nHeight ];
    isFree_sh[0][threadIdx.y+1][threadIdx.z+1]   = isFreeAll[ (t_j-1) + t_i*nWidth + t_k*nWidth*nHeight ];
  }
  //Right boundary
  if (t_j == nWidth-1){
    conc_sh  [blockDim.x+1][threadIdx.y+1][threadIdx.z+1]   =    concIn[ t_i*nWidth + t_k*nWidth*nHeight ];
    isFree_sh[blockDim.x+1][threadIdx.y+1][threadIdx.z+1]   = isFreeAll[ t_i*nWidth + t_k*nWidth*nHeight ];
    atomicAdd( &boundarySum , (float)conc_sh[blockDim.x][threadIdx.y+1][threadIdx.z+1] );
  }
  else if ( threadIdx.x == blockDim.x-1 ){
    conc_sh  [blockDim.x+1][threadIdx.y+1][threadIdx.z+1]   =    concIn[ (t_j+1) + t_i*nWidth + t_k*nWidth*nHeight ];
    isFree_sh[blockDim.x+1][threadIdx.y+1][threadIdx.z+1]   = isFreeAll[ (t_j+1) + t_i*nWidth + t_k*nWidth*nHeight ];
  }
  //Down boundary
  if (t_i == 0){
    conc_sh  [threadIdx.x+1][0][threadIdx.z+1]   =    concIn[ t_j + (nHeight-1)*nWidth + t_k*nWidth*nHeight ];
    isFree_sh[threadIdx.x+1][0][threadIdx.z+1]   = isFreeAll[ t_j + (nHeight-1)*nWidth + t_k*nWidth*nHeight ];
    atomicAdd( &boundarySum , (float)conc_sh[threadIdx.x+1][1][threadIdx.z+1] );
  }
  else if ( threadIdx.y == 0 ){
    conc_sh  [threadIdx.x+1][0][threadIdx.z+1]   =    concIn[ t_j + (t_i-1)*nWidth + t_k*nWidth*nHeight ];
    isFree_sh[threadIdx.x+1][0][threadIdx.z+1]   = isFreeAll[ t_j + (t_i-1)*nWidth + t_k*nWidth*nHeight ];
  }
  //Up boundary
  if (t_i == nHeight-1){
    conc_sh  [threadIdx.x+1][blockDim.y+1][threadIdx.z+1]   =    concIn[ t_j + t_k*nWidth*nHeight ];
    isFree_sh[threadIdx.x+1][blockDim.y+1][threadIdx.z+1]   = isFreeAll[ t_j + t_k*nWidth*nHeight ];
    atomicAdd( &boundarySum , (float)conc_sh[threadIdx.x+1][blockDim.y][threadIdx.z+1] );
  }
  else if ( threadIdx.y == blockDim.y-1 ){
    conc_sh  [threadIdx.x+1][blockDim.y+1][threadIdx.z+1]   =    concIn[ t_j + (t_i+1)*nWidth + t_k*nWidth*nHeight ];
    isFree_sh[threadIdx.x+1][blockDim.y+1][threadIdx.z+1]   = isFreeAll[ t_j + (t_i+1)*nWidth + t_k*nWidth*nHeight ];
  }
  //Bottom boundary
  if (t_k == 0){
    conc_sh  [threadIdx.x+1][threadIdx.y+1][0]   =    concIn[ t_j + t_i*nWidth + (nDepth-1)*nWidth*nHeight ];
    isFree_sh[threadIdx.x+1][threadIdx.y+1][0]   = isFreeAll[ t_j + t_i*nWidth + (nDepth-1)*nWidth*nHeight ];
    atomicAdd( &boundarySum , (float)conc_sh[threadIdx.x+1][threadIdx.y+1][1] );
  }
  else if ( threadIdx.z == 0 ){
    conc_sh  [threadIdx.x+1][threadIdx.y+1][0]   =    concIn[ t_j + t_i*nWidth + (t_k-1)*nWidth*nHeight ];
    isFree_sh[threadIdx.x+1][threadIdx.y+1][0]   = isFreeAll[ t_j + t_i*nWidth + (t_k-1)*nWidth*nHeight ];
  }
  //Top boundary
  if (t_k == nDepth-1){
    conc_sh  [threadIdx.x+1][threadIdx.y+1][blockDim.z+1]   =    concIn[ t_j + t_i*nWidth ];
    isFree_sh[threadIdx.x+1][threadIdx.y+1][blockDim.z+1]   = isFreeAll[ t_j + t_i*nWidth ];
    atomicAdd( &boundarySum , (float)conc_sh[threadIdx.x+1][threadIdx.y+1][blockDim.z] );
  }
  else if ( threadIdx.z == blockDim.z-1 ){
    conc_sh  [threadIdx.x+1][threadIdx.y+1][blockDim.z+1]   =    concIn[ t_j + t_i*nWidth + (t_k+1)*nWidth*nHeight ];
    isFree_sh[threadIdx.x+1][threadIdx.y+1][blockDim.z+1]   = isFreeAll[ t_j + t_i*nWidth + (t_k+1)*nWidth*nHeight ];
  }
  if (threadIdx.x == 0 and threadIdx.y == 0 and threadIdx.z == 0) blockBoundarySum[blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.y*gridDim.x] = boundarySum;
  __syncthreads();
  	 
//   cudaP oneThird = 1.0/3;
  

//   cudaP newConc = hx*conc_sh[threadIdx.x][threadIdx.y+1] + 
//     (1 - hx)*( conc_sh[threadIdx.x+2][threadIdx.y+1] + conc_sh[threadIdx.x+1][threadIdx.y] + conc_sh[threadIdx.x+1][threadIdx.y+2] )*oneThird +
//      ( hx*(1 - isFree_sh[threadIdx.x+2][threadIdx.y+1]) + 
//       (1 - hx)*( 3 - ( isFree_sh[threadIdx.x][threadIdx.y+1] + isFree_sh[threadIdx.x+1][threadIdx.y] + isFree_sh[threadIdx.x+1][threadIdx.y+2] ) )*oneThird )*conc_sh[threadIdx.x+1][threadIdx.y+1];
  
  cudaP newConc = hx*( conc_sh[threadIdx.x][threadIdx.y+1][threadIdx.z+1] + ( 1 - isFree_sh[threadIdx.x+2][threadIdx.y+1][threadIdx.z+1] )*conc_sh[threadIdx.x+1][threadIdx.y+1][threadIdx.z+1] ) +
    0.20*( 1 - hx )*( conc_sh[threadIdx.x+2][threadIdx.y+1][threadIdx.z+1] + conc_sh[threadIdx.x+1][threadIdx.y][threadIdx.z+1] + conc_sh[threadIdx.x+1][threadIdx.y+2][threadIdx.z+1] + conc_sh[threadIdx.x+1][threadIdx.y+1][threadIdx.z] + conc_sh[threadIdx.x+1][threadIdx.y+1][threadIdx.z+2] +
                      conc_sh[threadIdx.x+1][threadIdx.y+1][threadIdx.z+1]*( 5 - ( isFree_sh[threadIdx.x][threadIdx.y+1][threadIdx.z+1] + isFree_sh[threadIdx.x+1][threadIdx.y][threadIdx.z+1] + isFree_sh[threadIdx.x+1][threadIdx.y+2][threadIdx.z+1] + isFree_sh[threadIdx.x+1][threadIdx.y+1][threadIdx.z] + isFree_sh[threadIdx.x+1][threadIdx.y+1][threadIdx.z+2] ) ) );
    
  if ( isFree_sh[threadIdx.x+1][threadIdx.y+1][threadIdx.z+1] )  concentrationOut[tid] = newConc;
}
/////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////// 
__global__ void getCM_step1_kernel( const cudaP xMin, const cudaP yMin, const cudaP zMin, const cudaP dx, const cudaP dy, const cudaP dz, 
			     cudaP *concAll, cudaP *cm_xAll, cudaP *cm_yAll, cudaP *cm_zAll ){
  const int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  const int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  const int tid = t_j + t_i*blockDim.x*gridDim.x;

  cudaP x = t_j*dx + xMin;
  cudaP y = t_i*dy + yMin;
  cudaP conc = concAll[tid];
  cm_xAll[tid] = x*conc;
  cm_yAll[tid] = y*conc;
}
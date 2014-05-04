// #include <surface_functions.h>
#include <stdint.h>
#include <cuda.h>

texture<int, cudaTextureType3D, cudaReadModeElementType> tex_isFree;
texture<float, cudaTextureType3D, cudaReadModeElementType> tex_concentrationIn;
// surface< void, cudaSurfaceType3D> surf_concentrationOut;


// __global__ void countFreeNeighbors_kernel( const int nWidth, const int nHeight, const int nDepth, int *nFreeAll){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int t_k = blockIdx.z*blockDim.z + threadIdx.z;
//   int tid = t_j + t_i*nWidth + t_k*nWidth*blockDim.y*gridDim.y;
//   
//   unsigned char left   = tex3D( tex_isFree, t_j-1, t_i, t_k );
//   unsigned char right  = tex3D( tex_isFree, t_j+1, t_i, t_k );
//   unsigned char up     = tex3D( tex_isFree, t_j, t_i+1, t_k );
//   unsigned char down   = tex3D( tex_isFree, t_j, t_i-1, t_k );
//   unsigned char top    = tex3D( tex_isFree, t_j, t_i, t_k+1 );
//   unsigned char bottom = tex3D( tex_isFree, t_j, t_i, t_k-1 );
//   
//   //Set PERIODIC boundary conditions
//   if (t_i == 0)           down = tex3D( tex_isFree, t_j, nHeight-1, t_k );
//   if (t_i == (nHeight-1))   up = tex3D( tex_isFree, t_j, 0, t_k );
//   if (t_j == 0)           left = tex3D( tex_isFree, nWidth-1, t_i, t_k );
//   if (t_j == (nWidth-1)) right = tex3D( tex_isFree, 0, t_i, t_k );
//   if (t_k == 0)         bottom = tex3D( tex_isFree, t_j, t_i, nDepth-1 );
//   if (t_k == (nDepth-1))   top = tex3D( tex_isFree, t_j, t_i, 0 );
//   
//   int nFree = 0;
//   if ( left )   nFree += 1;
//   if ( right )  nFree += 1;
//   if ( down )   nFree += 1;
//   if ( up )     nFree += 1;
//   if ( top )    nFree += 1;
//   if ( bottom ) nFree += 1;
//   
//   nFreeAll[tid] = nFree;
// }
/////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////// 
__global__ void main_kernel_tex( const int nWidth, const int nHeight, const int nDepth,
				 int *isFreeAll, float *concentrationOut ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int t_k = blockIdx.z*blockDim.z + threadIdx.z;
  int tid = t_j + t_i*nWidth + t_k*nWidth*blockDim.y*gridDim.y;
  
  //Read neighbors occupancy 
  int left_isFree   = tex3D( tex_isFree, t_j-1, t_i, t_k );
  int right_isFree  = tex3D( tex_isFree, t_j+1, t_i, t_k );
  int up_isFree     = tex3D( tex_isFree, t_j, t_i+1, t_k );
  int down_isFree   = tex3D( tex_isFree, t_j, t_i-1, t_k );
  int top_isFree    = tex3D( tex_isFree, t_j, t_i, t_k+1 );
  int bottom_isFree = tex3D( tex_isFree, t_j, t_i, t_k-1 );
  //Set PERIODIC boundary conditions
  if (t_i == 0)           down_isFree = isFreeAll[ t_j + (nHeight-1)*nWidth + t_k*nWidth*nHeight ];
  if (t_i == (nHeight-1))   up_isFree = isFreeAll[ t_j + t_k*nWidth*nHeight ];
  if (t_j == 0)           left_isFree = isFreeAll[ (nWidth-1) + t_i*nWidth + t_k*nWidth*nHeight ];
  if (t_j == (nWidth-1)) right_isFree = isFreeAll[ t_i*nWidth + t_k*nWidth*nHeight ];
  if (t_k == 0)         bottom_isFree = isFreeAll[ t_j + t_i*nWidth + (nDepth-1)*nWidth*nHeight ];
  if (t_k == (nDepth-1))   top_isFree = isFreeAll[ t_j + t_i*nWidth ];
    
  //Read neighbors concentration
  float center_C = tex3D( tex_concentrationIn, t_j,   t_i, t_k );
  float left_C   = tex3D( tex_concentrationIn, t_j-1, t_i, t_k );
  float right_C  = tex3D( tex_concentrationIn, t_j+1, t_i, t_k );
  float up_C     = tex3D( tex_concentrationIn, t_j, t_i+1, t_k );
  float down_C   = tex3D( tex_concentrationIn, t_j, t_i-1, t_k );
  float top_C    = tex3D( tex_concentrationIn, t_j, t_i, t_k+1 );
  float bottom_C = tex3D( tex_concentrationIn, t_j, t_i, t_k-1 );
  //Set PERIODIC boundary conditions
  if (t_i == 0)           down_C = tex3D( tex_concentrationIn, t_j, nHeight-1, t_k );
  if (t_i == (nHeight-1))   up_C = tex3D( tex_concentrationIn, t_j, 0, t_k );
  if (t_j == 0)           left_C = tex3D( tex_concentrationIn, nWidth-1, t_i, t_k );
  if (t_j == (nWidth-1)) right_C = tex3D( tex_concentrationIn, 0, t_i, t_k );
  if (t_k == 0)         bottom_C = tex3D( tex_concentrationIn, t_j, t_i, nDepth-1 );
  if (t_k == (nDepth-1))   top_C = tex3D( tex_concentrationIn, t_j, t_i, 0 );
  
  float newConc;
  if ( isFreeAll[tid] ) newConc = ( ( left_C + right_C + down_C + up_C + bottom_C + top_C ) +
	( 6 - ( left_isFree + right_isFree + down_isFree + up_isFree + down_isFree + top_isFree ) )*center_C );
  else newConc = 0.f;
  
  concentrationOut[tid] = newConc/6;
//   else concentrationOut[tid] = 0.f;

}
  
  
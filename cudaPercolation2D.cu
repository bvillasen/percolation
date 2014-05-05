// #include <pycuda-complex.hpp>
// #include <surface_functions.h>
#include <stdint.h>
#include <cuda.h>

typedef unsigned char uchar;

texture< int, cudaTextureType2D, cudaReadModeElementType> tex_isFree;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_concentrationIn;
/////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////// 
__global__ void main_kernel_tex( const int nWidth, const int nHeight, float hx, int *isFreeAll,
			     float *concentrationOut ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_j + t_i*blockDim.x*gridDim.x;
   
  //Read neighbors occupancy
  int left_isFree   =  tex2D( tex_isFree, t_j-1, t_i ); ; 
  int right_isFree  =  tex2D( tex_isFree, t_j+1, t_i ); ; 
  int up_isFree     =  tex2D( tex_isFree, t_j, t_i+1 ); ; 
  int down_isFree   =  tex2D( tex_isFree, t_j, t_i-1 ); ; 
  //Set PERIODIC boundary conditions
  if (t_i == 0)           down_isFree = isFreeAll[ t_j + (nHeight-1)*nWidth ];
  if (t_i == (nHeight-1))   up_isFree = isFreeAll[ t_j ];
  if (t_j == 0)           left_isFree = isFreeAll[ (nWidth-1) + (t_i)*nWidth ];
  if (t_j == (nWidth-1)) right_isFree = isFreeAll[ (t_i)*nWidth ];

  //Read neighbors concentration
  float center_C = tex2D( tex_concentrationIn, t_j,   t_i );
  float left_C   = tex2D( tex_concentrationIn, t_j-1, t_i );
  float right_C  = tex2D( tex_concentrationIn, t_j+1, t_i );
  float up_C     = tex2D( tex_concentrationIn, t_j, t_i+1 );
  float down_C   = tex2D( tex_concentrationIn, t_j, t_i-1 );
  //Set PERIODIC boundary conditions
  if (t_i == 0)           down_C = tex2D( tex_concentrationIn, t_j, nHeight-1 );
  if (t_i == (nHeight-1))   up_C = tex2D( tex_concentrationIn, t_j, 0 );
  if (t_j == 0)           left_C = tex2D( tex_concentrationIn, nWidth-1, t_i );
  if (t_j == (nWidth-1)) right_C = tex2D( tex_concentrationIn, 0, t_i );
 
  float newConcentration = 0.25f*( left_C + right_C + down_C + up_C ) +
         0.25f*( 4 - ( left_isFree + right_isFree + down_isFree + up_isFree ) )*center_C;

//   float newConcentration = hx*left_C + (1.f - hx)*(right_C + down_C + up_C )/3.f +
//       ( hx*(1 - right_isFree) + (1.f-hx)*( 3 - ( left_isFree + down_isFree + up_isFree ) )/3.f )*center_C;
//       
  if ( isFreeAll[tid] )  concentrationOut[tid] = newConcentration;
}
/////////////////////////////////////////////////////////////////////////////////////// 
/////////////////////////////////////////////////////////////////////////////////////// 
__global__ void main_kernel_shared( const int nWidth, const int nHeight, cudaP hx, uchar *isFreeAll,
			          cudaP *concIn, cudaP *concentrationOut ){
  const int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  const int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  const int tid = t_j + t_i*blockDim.x*gridDim.x;
  
  //Read my neighbors concentration
  __shared__ uchar   isFree_sh[ %(B_WIDTH)s + 2 ][ %(B_HEIGHT)s + 2 ];
  __shared__ cudaP conc_sh[ %(B_WIDTH)s + 2 ][ %(B_HEIGHT)s + 2 ];
  conc_sh[threadIdx.x+1][threadIdx.y+1] =    concIn[tid] ;
  isFree_sh[threadIdx.x+1][threadIdx.y+1] = isFreeAll[tid];
  //Left boundary
  if (t_j == 0){
    conc_sh[0][threadIdx.y+1] =    concIn[ (nWidth-1) + t_i*nWidth ];
    isFree_sh[0][threadIdx.y+1] = isFreeAll[ (nWidth-1) + t_i*nWidth ];
  }
  else if ( threadIdx.x == 0 ){
    conc_sh[0][threadIdx.y+1] =    concIn[ (t_j-1) + t_i*nWidth ];
    isFree_sh[0][threadIdx.y+1] = isFreeAll[ (t_j-1) + t_i*nWidth ];
  }
  //Right boundary
  if (t_j == nWidth-1){
    conc_sh[blockDim.x+1][threadIdx.y+1] =    concIn[ t_i*nWidth ];
    isFree_sh[blockDim.x+1][threadIdx.y+1] = isFreeAll[ t_i*nWidth ];
  }
  else if ( threadIdx.x == blockDim.x-1 ){
    conc_sh[blockDim.x+1][threadIdx.y+1] =    concIn[ (t_j+1) + t_i*nWidth ];
    isFree_sh[blockDim.x+1][threadIdx.y+1] = isFreeAll[ (t_j+1) + t_i*nWidth ];
  }
  //Down boundary
  if (t_i == 0){
    conc_sh[threadIdx.x+1][0] =    concIn[ t_j + (nHeight-1)*nWidth ];
    isFree_sh[threadIdx.x+1][0] = isFreeAll[ t_j + (nHeight-1)*nWidth ];
  }
  else if ( threadIdx.y == 0 ){
    conc_sh[threadIdx.x+1][0] =    concIn[ t_j + (t_i-1)*nWidth ];
    isFree_sh[threadIdx.x+1][0] = isFreeAll[ t_j + (t_i-1)*nWidth ];
  }
  //Up boundary
  if (t_i == nHeight-1){
    conc_sh[threadIdx.x+1][blockDim.y+1] =    concIn[ t_j ];
    isFree_sh[threadIdx.x+1][blockDim.y+1] = isFreeAll[ t_j ];
  }
  else if ( threadIdx.y == blockDim.y-1 ){
    conc_sh[threadIdx.x+1][blockDim.y+1] =    concIn[ t_j + (t_i+1)*nWidth ];
    isFree_sh[threadIdx.x+1][blockDim.y+1] = isFreeAll[ t_j + (t_i+1)*nWidth ];
  }
  __syncthreads();
  	 
  cudaP oneThird = 1.0/3;
  
//   cudaP newConc = hx*conc_sh[threadIdx.x][threadIdx.y+1] + 
//     (1 - hx)*( conc_sh[threadIdx.x+2][threadIdx.y+1] + conc_sh[threadIdx.x+1][threadIdx.y] + conc_sh[threadIdx.x+1][threadIdx.y+2] )*oneThird +
//     ( hx*(1 - isFree_sh[threadIdx.x+2][threadIdx.y+1]) + 
//       (1 - hx)*( 3 - ( isFree_sh[threadIdx.x][threadIdx.y+1] + isFree_sh[threadIdx.x+1][threadIdx.y] + isFree_sh[threadIdx.x+1][threadIdx.y+2] ) )*oneThird )*conc_sh[threadIdx.x+1][threadIdx.y+1];

  cudaP newConc = hx*( conc_sh[threadIdx.x][threadIdx.y+1] + ( 1 - isFree_sh[threadIdx.x+2][threadIdx.y+1] )*conc_sh[threadIdx.x+1][threadIdx.y+1] ) +
    oneThird*( 1 - hx )*( conc_sh[threadIdx.x+2][threadIdx.y+1] + conc_sh[threadIdx.x+1][threadIdx.y] + conc_sh[threadIdx.x+1][threadIdx.y+2] +
                         conc_sh[threadIdx.x+1][threadIdx.y+1]*( 3 - ( isFree_sh[threadIdx.x][threadIdx.y+1] + isFree_sh[threadIdx.x+1][threadIdx.y] + isFree_sh[threadIdx.x+1][threadIdx.y+2] ) ) );
    
  if ( isFree_sh[threadIdx.x+1][threadIdx.y+1] )  concentrationOut[tid] = newConc;
}
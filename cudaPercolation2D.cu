// #include <pycuda-complex.hpp>
// #include <surface_functions.h>
#include <stdint.h>
#include <cuda.h>

typedef unsigned char uchar;

texture< int, cudaTextureType2D, cudaReadModeElementType> tex_isFree;
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_concentrationIn;
// surface< void, cudaSurfaceType2D> surf_concentrationOut;

// __global__ void countFreeNeighbors_kernel( const int nWidth, const int nHeight, int *nFreeAll){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int tid = t_j + t_i*blockDim.x*gridDim.x;
//   
//   uchar left   = tex2D( tex_isFree, t_j-1, t_i );
//   uchar right  = tex2D( tex_isFree, t_j+1, t_i );
//   uchar up     = tex2D( tex_isFree, t_j, t_i+1 );
//   uchar down   = tex2D( tex_isFree, t_j, t_i-1 );
//   
//   //Set PERIODIC boundary conditions
//   if (t_i == 0)           down = tex2D( tex_isFree, t_j, nHeight-1 );
//   if (t_i == (nHeight-1))   up = tex2D( tex_isFree, t_j, 0 );
//   if (t_j == 0)           left = tex2D( tex_isFree, nWidth-1, t_i );
//   if (t_j == (nWidth-1)) right = tex2D( tex_isFree, 0, t_i );
//   
//   int nFree = 0;
//   if ( left )  nFree += 1;
//   if ( right ) nFree += 1;
//   if ( down )  nFree += 1;
//   if ( up )    nFree += 1;
// 
//   nFreeAll[tid] = nFree;
// }
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
__global__ void main_kernel_shared( const int nWidth, const int nHeight, cudaP hx, int *isFreeAll,
			          cudaP *concIn, cudaP *concentrationOut ){
  const int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  const int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  const int tid = t_j + t_i*blockDim.x*gridDim.x;
  
  //Read my neighbors concentration
  __shared__ int   isFree_sh[ %(B_WIDTH)s + 2 ][ %(B_HEIGHT)s + 2 ];
  __shared__ cudaP concIn_sh[ %(B_WIDTH)s + 2 ][ %(B_HEIGHT)s + 2 ];
  concIn_sh[threadIdx.x+1][threadIdx.y+1] =    concIn[tid] ;
  isFree_sh[threadIdx.x+1][threadIdx.y+1] = isFreeAll[tid];
  //Left boundary
  if (t_j == 0){
    concIn_sh[0][threadIdx.y+1] =    concIn[ (nWidth-1) + t_i*nWidth ];
    isFree_sh[0][threadIdx.y+1] = isFreeAll[ (nWidth-1) + t_i*nWidth ];
  }
  else if ( threadIdx.x == 0 ){
    concIn_sh[0][threadIdx.y+1] =    concIn[ (t_j-1) + t_i*nWidth ];
    isFree_sh[0][threadIdx.y+1] = isFreeAll[ (t_j-1) + t_i*nWidth ];
  }
  //Right boundary
  if (t_j == nWidth-1){
    concIn_sh[blockDim.x+1][threadIdx.y+1] =    concIn[ t_i*nWidth ];
    isFree_sh[blockDim.x+1][threadIdx.y+1] = isFreeAll[ t_i*nWidth ];
  }
  else if ( threadIdx.x == blockDim.x-1 ){
    concIn_sh[blockDim.x+1][threadIdx.y+1] =    concIn[ (t_j+1) + t_i*nWidth ];
    isFree_sh[blockDim.x+1][threadIdx.y+1] = isFreeAll[ (t_j+1) + t_i*nWidth ];
  }
  //Down boundary
  if (t_i == 0){
    concIn_sh[threadIdx.x+1][0] =    concIn[ t_j + (nHeight-1)*nWidth ];
    isFree_sh[threadIdx.x+1][0] = isFreeAll[ t_j + (nHeight-1)*nWidth ];
  }
  else if ( threadIdx.y == 0 ){
    concIn_sh[threadIdx.x+1][0] =    concIn[ t_j + (t_i-1)*nWidth ];
    isFree_sh[threadIdx.x+1][0] = isFreeAll[ t_j + (t_i-1)*nWidth ];
  }
  //Up boundary
  if (t_i == nHeight-1){
    concIn_sh[threadIdx.x+1][blockDim.y+1] =    concIn[ t_j ];
    isFree_sh[threadIdx.x+1][blockDim.y+1] = isFreeAll[ t_j ];
  }
  else if ( threadIdx.y == blockDim.y-1 ){
    concIn_sh[threadIdx.x+1][blockDim.y+1] =    concIn[ t_j + (t_i+1)*nWidth ];
    isFree_sh[threadIdx.x+1][blockDim.y+1] = isFreeAll[ t_j + (t_i+1)*nWidth ];
  }
  __syncthreads();
  
  cudaP newConc = 0.25*( concIn_sh[threadIdx.x][threadIdx.y+1] + concIn_sh[threadIdx.x+2][threadIdx.y+1] +
                          concIn_sh[threadIdx.x+1][threadIdx.y] + concIn_sh[threadIdx.x+1][threadIdx.y+2] )+
         0.25*( 4 - ( isFree_sh[threadIdx.x][threadIdx.y+1] + isFree_sh[threadIdx.x+2][threadIdx.y+1] + 
                      isFree_sh[threadIdx.x+1][threadIdx.y] + isFree_sh[threadIdx.x+1][threadIdx.y+2] ) )*concIn_sh[threadIdx.x+1][threadIdx.y+1];
	 
	 
  
//   cudaP newConc = hx*concIn_sh[threadIdx.x][threadIdx.y+1] + 
//     (1 - hx)*( concIn_sh[threadIdx.x+2][threadIdx.y+1] + concIn_sh[threadIdx.x+1][threadIdx.y] + concIn_sh[threadIdx.x+1][threadIdx.y+2] )/3 +
//     ( hx*(1 - right_isFree) + (1 - hx)*( 3 - ( left_isFree + down_isFree + up_isFree ) )/3 )*concIn_sh[threadIdx.x+1][threadIdx.y+1];
	 
//   cudaP newConcentration = hx*left_C + (1.f - hx)*(right_C + down_C + up_C )/3.f +
//     hx*(1 - right_isFree)*center_C + (1.f-hx)*( 3 - ( left_isFree + down_isFree + up_isFree ) )/3.f*center_C;
 
  if ( isFree_sh[threadIdx.x+1][threadIdx.y+1] )  concentrationOut[tid] = newConc;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// __global__ void findActivity_kernel( cudaP minVal, cudaP *concentration, uchar *activeBlocks ){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int tid = t_j + t_i*blockDim.x*gridDim.x;
//   int tid_b = threadIdx.x + threadIdx.y*blockDim.x;
// 
//   __shared__ cudaP concentration_sh[ %(THREADS_PER_BLOCK)s ];
//   concentration_sh[tid_b] = concentration[tid];
//   __syncthreads();
//   
//   int i = blockDim.x*blockDim.y / 2;
//   while ( i > 0 ){
//     if ( tid_b < i ) concentration_sh[tid_b] = concentration_sh[tid_b] + concentration_sh[tid_b+i];
//     __syncthreads();
//     i /= 2;
//   }
//   if ( tid_b == 0 ){
//     if (concentration_sh[0] >= minVal ) {
//       activeBlocks[ blockIdx.x + blockIdx.y*gridDim.x ] = (uchar) 1;
//       //right 
//       if (blockIdx.x < gridDim.x-1) activeBlocks[ (blockIdx.x+1) + blockIdx.y*gridDim.x ] = (uchar) 1;
//       //left
//       if (blockIdx.x > 0) activeBlocks[ (blockIdx.x-1) + blockIdx.y*gridDim.x ] = (uchar) 1;
//       //up 
//       if (blockIdx.y < gridDim.y-1) activeBlocks[ blockIdx.x + (blockIdx.y+1)*gridDim.x  ] = (uchar) 1;
//       //Down 
//       if (blockIdx.y > 0) activeBlocks[ blockIdx.x + (blockIdx.y-1)*gridDim.x  ] = (uchar) 1;
//     }
//   }
// }
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// __global__ void getActivity_kernel(  uchar *activeBlocks, uchar *activeThreads ){
//   int t_j = blockIdx.x*blockDim.x + threadIdx.x;
//   int t_i = blockIdx.y*blockDim.y + threadIdx.y;
//   int tid = t_j + t_i*blockDim.x*gridDim.x ;
//   int tid_b = threadIdx.x + threadIdx.y*blockDim.x;
//   int bid = blockIdx.x + blockIdx.y*gridDim.x;
//   
//   __shared__ uchar activeBlock;
//   if (tid_b == 0 ) activeBlock = activeBlocks[bid];
//   __syncthreads();
//   
//   if ( activeBlock ) activeThreads[tid] = (uchar) 1;
//   else activeThreads[tid] = (uchar) 0;
// }

/*
__global__ void main_kernel( const int nWidth, const int nHeight, int *nFreeAll, uchar *isFree, cudaP *concentrationIn, cudaP *concentrationOut ){
  int t_j = blockIdx.x*blockDim.x + threadIdx.x;
  int t_i = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_j + t_i*blockDim.x*gridDim.x;
  
  cudaP localConc = concentrationIn[tid];
  
  __shared__ uchar isFree_sh[ %(B_WIDTH)s + 2 ][ %(B_HEIGHT)s + 2 ];
  __shared__ cudaP concentrationIn_sh[ %(B_WIDTH)s + 2 ][ %(B_HEIGHT)s + 2 ];
  __shared__ cudaP concentrationOut_sh[ %(B_WIDTH)s + 2 ][ %(B_HEIGHT)s + 2 ];
  isFree_sh[threadIdx.x+1][threadIdx.y+1] = isFree[tid]; 
  concentrationIn_sh[threadIdx.x+1][threadIdx.y+1] = localConc ;
  concentrationOut_sh[threadIdx.x+1][threadIdx.y+1] = cudaP(0.);
  
  //Left boundary
  if (t_j == 0){  //Set periodic boundary
    isFree_sh[0][threadIdx.y+1] = isFree[ (nWidth-1) + t_i*nWidth ];
    concentrationIn_sh[0][threadIdx.y+1] = concentrationIn[ (nWidth-1) + t_i*nWidth ];
    concentrationOut_sh[0][threadIdx.y+1] = cudaP(0.);
  }
  else if ( threadIdx.x == 0 ){
    isFree_sh[0][threadIdx.y+1] = isFree[ (t_j-1) + t_i*nWidth ];
    concentrationIn_sh[0][threadIdx.y+1] = concentrationIn[ (t_j-1) + t_i*nWidth ];
    concentrationOut_sh[0][threadIdx.y+1] = cudaP(0.);
  }
  //Right boundary
  if (t_j == nWidth-1){  //Set periodic boundary
    isFree_sh[blockDim.x+1][threadIdx.y+1] = isFree[ t_i*nWidth ];
    concentrationIn_sh[blockDim.x+1][threadIdx.y+1] = concentrationIn[ t_i*nWidth ];
    concentrationOut_sh[blockDim.x+1][threadIdx.y+1] = cudaP(0.);
  }
  else if ( threadIdx.x == blockDim.x-1 ){
    isFree_sh[blockDim.x+1][threadIdx.y+1] = isFree[ (t_j+1) + t_i*nWidth ];
    concentrationIn_sh[blockDim.x+1][threadIdx.y+1] = concentrationIn[ (t_j+1) + t_i*nWidth ];
    concentrationOut_sh[blockDim.x+1][threadIdx.y+1] = cudaP(0.);
  }
  //Down boundary
  if (t_i == 0){  //Set periodic boundary
    isFree_sh[threadIdx.x+1][0] = isFree[ t_j + (nHeight-1)*nWidth ];
    concentrationIn_sh[threadIdx.x+1][0] = concentrationIn[ t_j + (nHeight-1)*nWidth ];
    concentrationOut_sh[threadIdx.x+1][0] = cudaP(0.);
  }
  else if ( threadIdx.y == 0 ){
    isFree_sh[threadIdx.x+1][0] = isFree[ t_j + (t_i-1)*nWidth ];
    concentrationIn_sh[threadIdx.x+1][0] = concentrationIn[ t_j + (t_i-1)*nWidth ];
    concentrationOut_sh[threadIdx.x+1][0] = cudaP(0.);
  }
  //Up boundary
  if (t_i == nHeight-1){  //Set periodic boundary
    isFree_sh[threadIdx.x+1][blockDim.y+1] = isFree[ t_j ];
    concentrationIn_sh[threadIdx.x+1][blockDim.y+1] = concentrationIn[ t_j ];
    concentrationOut_sh[threadIdx.x+1][blockDim.y+1] = cudaP(0.);
  }
  else if ( threadIdx.y == blockDim.y-1 ){
    isFree_sh[threadIdx.x+1][blockDim.y+1] = isFree[ t_j + (t_i+1)*nWidth ];
    concentrationIn_sh[threadIdx.x+1][blockDim.y+1] = concentrationIn[ t_j + (t_i+1)*nWidth ];
    concentrationOut_sh[threadIdx.x+1][blockDim.y+1] = cudaP(0.);
  }
  __syncthreads();
  
  int nFree = nFreeAll[tid];
//   concentrationOut[tid] = nFree;
  cudaP sendVal = localConc/nFree;
  if (isFree_sh[threadIdx.x][threadIdx.y + 1])     atomicAdd( &(concentrationOut_sh[threadIdx.x][threadIdx.y + 1]) , sendVal);      //Left 	
  if (isFree_sh[threadIdx.x + 2][threadIdx.y + 1]) atomicAdd( &(concentrationOut_sh[threadIdx.x + 2][threadIdx.y + 1]) , sendVal);  //Right
  if (isFree_sh[threadIdx.x + 1][threadIdx.y])     atomicAdd( &(concentrationOut_sh[threadIdx.x + 1][threadIdx.y]) , sendVal);      //Down
  if (isFree_sh[threadIdx.x + 1][threadIdx.y + 2]) atomicAdd( &(concentrationOut_sh[threadIdx.x + 1][threadIdx.y + 2]) , sendVal);  //Up
  __syncthreads();
//   if ( isFree_sh[threadIdx.x + 1][threadIdx.y + 1] ){
  //Write shared memory to global
  concentrationOut[tid] = concentrationOut_sh[threadIdx.x+1][threadIdx.y+1];
  //Write shared memory boundary
  //Left boundary
  if (t_j == 0)                atomicAdd( &(concentrationOut[ (nWidth-1) + t_i*nWidth ]), concentrationOut_sh[0][threadIdx.y + 1] );
  else if ( threadIdx.x == 0 ) atomicAdd( &(concentrationOut[ (t_j-1) + t_i*nWidth ]), concentrationOut_sh[0][threadIdx.y + 1] );
  //Rigth boundary
  if (t_j == nWidth-1)                    atomicAdd( &(concentrationOut[ t_i*nWidth ]), concentrationOut_sh[blockDim.x + 1][threadIdx.y + 1] );
  else if ( threadIdx.x == blockDim.x-1 ) atomicAdd( &(concentrationOut[ (t_j+1) + t_i*nWidth ]), concentrationOut_sh[blockDim.x + 1][threadIdx.y + 1] );
  //Down boundary
  if (t_i == 0)                atomicAdd( &(concentrationOut[ t_j + (nHeight-1)*nWidth ]), concentrationOut_sh[threadIdx.x + 1][0] );
  else if ( threadIdx.y == 0 ) atomicAdd( &(concentrationOut[ t_j + (t_i-1)*nWidth ]), concentrationOut_sh[threadIdx.x + 1][0] );
  //Up boundary
  if (t_i == nHeight-1)                    atomicAdd( &(concentrationOut[ t_j ]), concentrationOut_sh[threadIdx.x + 1][ blockDim.y + 1] );
  else if ( threadIdx.y == blockDim.y-1 )  atomicAdd( &(concentrationOut[ t_j + (t_i+1)*nWidth ]), concentrationOut_sh[threadIdx.x + 1][blockDim.y + 1] );
  
}*/
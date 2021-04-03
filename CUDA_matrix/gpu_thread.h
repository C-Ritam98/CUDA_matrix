#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(gran, start, end) std::chrono::duration_cast<gran>(end - start).count()

__global__ void GPUDriverFunction(int * A, int * B, int N)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    A[row * N + col] = A[row * N + col] * B[col * N + N - 1 - row];

}


__global__ void GPUDriverFunction_2(int * A,int* output,int N)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if( row < 1 && col < 2*N-1 )
  {
    int diagLength = col+1;
                int rowA = 0,colA = 0;

                if(col>=N)
                {
                        diagLength = 2*N - 1 - col;
                       rowA = col - N + 1;
                }

               colA = col - rowA;
               int temp = 0;


                while(diagLength--) {
                        temp += A[N * rowA + colA];
                        rowA++; colA--;
                }
                output[col] = temp;
  }
}


void gpuThread(int N, int *matA, int *matB, int *output)
{
    int size = N * N * sizeof(int);
    int BLOCK_SIZE = N>=16?16:N; // block_size = min(16,N)
    int* d_A ;
//................................ section of code performing elementwise multiplication of matA and matB.............................................................................

    // allocation of space for matA in device
    cudaMalloc((void **)&d_A, size);
    cudaError_t result = cudaMemcpy(d_A, matA, size, cudaMemcpyHostToDevice);
    if(result != cudaSuccess)
    {
     cout << "Not copied matA from Host to Device!!" << endl;
    }
    
    // allocation of space for matB in device
    int* d_B;
    cudaMalloc((void **)&d_B, size);
    result = cudaMemcpy(d_B, matB, size,cudaMemcpyHostToDevice);
    if(result != cudaSuccess)
    {
     cout << "Not copied matB from Host to Device!!" << endl;
    }
 

    auto begin = TIME_NOW;
    
    // defining the block and grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    GPUDriverFunction<<<dimGrid, dimBlock>>>(d_A, d_B, N);   // call the functions for calculating the matrix A in GPU
    auto end = TIME_NOW;
    cout << "GPU execution time part 1: " << (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n";

   // free unnecessary space allocated to matB in Device
   cudaFree(d_B);

//.................................section of code handling the diagonal manipulation of the matrix A..............................................................................

   int *d_output;
   cudaMalloc((void **)&d_output, (2*N-1)*sizeof(int));  // allocate space for *output* array in Device

   // defining block and grid dimensions
   dim3 dimBlock_2(BLOCK_SIZE,1);
   dim3 dimGrid_2(2*N/BLOCK_SIZE , 1);
   begin = TIME_NOW;
   GPUDriverFunction_2<<< dimGrid_2, dimBlock_2 >>>(d_A,d_output,N); // invoke the function for adding the elements of matrix A, diagonally in GPU
   end = TIME_NOW;
   cout << "GPU execution time part 2: " << (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n";

   result = cudaMemcpy(output, d_output,(2*N-1)*sizeof(int),cudaMemcpyDeviceToHost);  // copying results back to Host

   if(result != cudaSuccess)
   {
     cout<<"Device to Host mem copy not done!"<<endl;
   }
    
    
    // free unnecessary space 
    cudaFree(d_A);
    cudaFree(d_output);
}

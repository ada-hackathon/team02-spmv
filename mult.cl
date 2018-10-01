__kernel
void mult(__global const float* A,__global const float* B,__global  float* C,const int cols){
                      int i = get_global_id(0);
                      int j = get_global_id(1);
                       int k ;
                        float temp = 0;

                       for(k = 0;k<cols;k++){
                             temp+= A[i*cols+k] * B[k*cols + j ]; }

                         C[i*cols+j]+=temp;};


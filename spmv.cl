#include "spmv.h"

// void ellpack(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N])
// {
//     int i, j;
//     TYPE Si;
// 
//     ellpack_1 : for (i=0; i<N; i++) {
//         TYPE sum = out[i];
//         ellpack_2 : for (j=0; j<L; j++) {
//                 Si = nzval[j + i*L] * vec[cols[j + i*L]];
//                 sum += Si;
//         }
//         out[i] = sum;
//     }
// }

__kernel void ellpack(__global const TYPE nzval[N*L], __global const int32_t cols[N*L], __global const TYPE vec[N], __global TYPE out[N]) {
    int i = get_global_id(0);
    TYPE sum = out[i];
    for (int j = 0; j < L; j++) {
             TYPE Si = nzval[j + i*L] * vec[cols[j + i*L]];
             sum += Si;
    }
    out [i] = sum;
}

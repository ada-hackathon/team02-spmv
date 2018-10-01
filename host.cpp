/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/
//OpenCL utility layer include
#include "xcl2.hpp"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

#include "support.h"
#include "spmv.h"

void ellpack_cpu(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N])
{
    int i, j;
    TYPE Si;

    ellpack_1 : for (i=0; i<N; i++) {
        TYPE sum = out[i];
        ellpack_2 : for (j=0; j<L; j++) {
                Si = nzval[j + i*L] * vec[cols[j + i*L]];
                sum += Si;
        }
        out[i] = sum;
    }
}
using namespace std;
int main(int argc, char** argv)
{
    /*
     * Allocate Memory for spmv
     */
    char *in_file;
    char *check_file;
    in_file = "input.data";
    check_file = "check.data";
    int in_fd;
    char *data;
    data = (char*)malloc(INPUT_SIZE);
    assert( data!=NULL && "Out of memory" );
    in_fd = open( in_file, O_RDONLY );
    assert( in_fd>0 && "Couldn't open input data file");
    input_to_data(in_fd, data);
    struct bench_args_t *clinput = (struct bench_args_t *)data;
    fprintf(stderr, "Read from input.data\n");
    /*
     * End of init
     */
    //OPENCL HOST CODE AREA START
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    cl::Context context(device);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);
    std::string device_name = device.getInfo<CL_DEVICE_NAME>(); 

    //Create Program and Kernel
    std::string binaryFile = xcl::find_binary_file(device_name,"spmv");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);
    cl::Program program(context, devices, bins);
    cl::Kernel krnl_mult(program,"ellpack");

    //Allocate Buffer in Global Memory
    cl::Buffer buffer_input1 (context, CL_MEM_READ_ONLY,
            N*L*sizeof(TYPE));
    cl::Buffer buffer_input2 (context, CL_MEM_READ_ONLY,
            N*L*sizeof(int32_t));
    cl::Buffer buffer_input3 (context, CL_MEM_READ_ONLY, 
            N*sizeof(TYPE));
    cl::Buffer buffer_input4(context, CL_MEM_READ_WRITE, 
            N*sizeof(TYPE));

    //Copy input data to device global memory
    q.enqueueWriteBuffer(buffer_input1, CL_TRUE, 0, N*L*sizeof(TYPE), clinput->nzval);
    q.enqueueWriteBuffer(buffer_input2, CL_TRUE, 0, N*L*sizeof(int32_t), clinput->cols);
    q.enqueueWriteBuffer(buffer_input3, CL_TRUE, 0, N*sizeof(TYPE), clinput->vec);
    q.enqueueWriteBuffer(buffer_input4, CL_TRUE, 0, N*sizeof(TYPE), clinput->out);

    //Set the Kernel Arguments
    int narg=0;
    krnl_mult.setArg(narg++,buffer_input1);
    krnl_mult.setArg(narg++,buffer_input2);
    krnl_mult.setArg(narg++,buffer_input3);
    krnl_mult.setArg(narg++,buffer_input4);

    //Launch the Kernel
    fprintf(stderr, "Launching Kernel\n");
    q.enqueueNDRangeKernel(krnl_mult,cl::NullRange,cl::NDRange(N),cl::NullRange);

    //Copy Result from Device Global Memory to Host Local Memory
    q.enqueueReadBuffer(buffer_input4, CL_TRUE, 0, N*sizeof(TYPE), clinput->out);

    q.finish();

    //OPENCL HOST CODE AREA END
    /*
     * Copy Back and Check
     */
    int out_fd;
    out_fd = open("output.data", O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
    assert( out_fd>0 && "Couldn't open output data file" );
    data_to_output(out_fd, data);
    close(out_fd);

    int check_fd;
    char *ref;
    ref = (char*)malloc(INPUT_SIZE);
    assert( ref!=NULL && "Out of memory" );
    check_fd = open( check_file, O_RDONLY );
    assert( check_fd>0 && "Couldn't open check data file");
    fprintf(stderr, "Writing to output.data\n");
    output_to_data(check_fd, ref);
    fprintf(stderr, "Checking\n");
    if( !check_data(data, ref) ) {
        fprintf(stderr, "Benchmark results are incorrect\n");
        return -1;
    }
    else {
        fprintf(stderr, "OPENCL GOOD\n");
    }
    free(data);
    free(ref);
    close(check_fd);
    return 0;
}

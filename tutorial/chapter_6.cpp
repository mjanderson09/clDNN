/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <../api/CPP/cldnn_defs.h>
#include <../api/CPP/engine.hpp>
#include <../api/CPP/input_layout.hpp>
#include <../api/CPP/memory.hpp>
#include <../api/CPP/data.hpp>
#include <../api/CPP/topology.hpp>
#include <../api/CPP/network.hpp>
#include <../api/CPP/convolution.hpp>
#include <iostream>
#include <chrono>
#include <tchar.h>
#include <windows.h>

#include "helper_functions.h"

/*! @page c6 How to add my own kernel implementation.
* @section intro Introduction
* In this chapter we will learn how to add a new Convolution kernel into clDNN kernel selector.
* 
* Please take a look in the files:
*   "convolution_kernel_tutorial.cpp"
*   "convolution_kernel_tutorial.h"
*   "convolution_tutorial.cl"
*
* @include chapter_6.cpp
*
*
*/

using namespace cldnn;

void benchmark_conv(network& network, const memory& input_memory)
{
    // Set/update network input
    network.set_input_data("conv_input", input_memory);

    // Start network execution
    auto outputs = network.execute();
}

void chapter_6(engine& engine, int N, int C, int H, int W, int K, int R, int S, int niter, int s, cldnn::format input_format, cldnn::format weights_format, int P, int Q)
{
    std::cout << std::endl << "-- Chapter 6 --" << std::endl;

    // We are going to implement a simple network with Convolution layer:
    //      input:          227x227 with 3 feature maps
    //      filter size:    3x3
    //      stride:         1,1
    //      offset:         0,0
    //
    // We use this code as an helper to test our new convolution kernel

    // Create input memory for convolution layer
    memory input_prim = memory::allocate(engine, { data_types::f32, input_format, {spatial(H,W), batch(N), feature(C)} });
    memory weights    = memory::allocate(engine, { data_types::f32, weights_format, {spatial(R,S), batch(K), feature(C)} } );
    memory bias = memory::allocate(engine, { data_types::f32, format::bfyx, spatial(K)  });
    tensor stride(1,1,s,s);
    tensor output_size(N,K,Q,P);

    set_values(input_prim, get_simple_data<float>(input_prim));
    set_values(weights,    get_simple_data<float>(weights));
    set_values(bias,    get_simple_data<float>(bias));

    // Create a topology with a simple Convolution layer
    topology topology(
        input_layout("conv_input", input_prim.get_layout()),
        data("conv_weights", weights),
        data("conv_bias", bias),
        convolution(
            "conv",
            "conv_input",
            { "conv_weights" },
            { "conv_bias" },
            stride,
            {0,0,0,0},
            {1,1,1,1},
            false,
            0.0f,
            output_size
            )
    );


    build_options build_opt;
    // Optimize_data flag can change weights and outputs layouts. Let take a look at 
    build_opt.set_option(build_option::outputs(topology.get_primitive_ids()));
    // Set option to optimize data.
    build_opt.set_option(build_option::optimize_data(true));

    network network(engine, topology, build_opt);

    // Set input.
    network.set_input_data("conv_input", input_prim);
    // Ready to go.
    auto outputs = network.execute();
    auto output = outputs.at("conv").get_memory();
    __int64 start = 0;
    __int64 end = 0;
    __int64 freq = 0;

    QueryPerformanceCounter((LARGE_INTEGER *)&start);

    for(int iter = 0 ; iter < niter ; iter++)
    {
      benchmark_conv(network, input_prim);
    }
    output = outputs.at("conv").get_memory();
    // Get direct access to output memory
    cldnn::pointer<float> out_ptr(output);
    QueryPerformanceCounter((LARGE_INTEGER *)&end);

    // Analyze result
    double avg = 0.0;
    int cnt = 0;
    for(auto it = out_ptr.begin() ; it != out_ptr.end() ; it++)
    {
      avg += (double)(*it);
      cnt++;
    }

    // We have table of profiling metrics.
    QueryPerformanceFrequency((LARGE_INTEGER *) &freq);
    double runtime = ((end-start) * 1.0 / freq) / (double)niter;

    std::cout << "PERFDUMP" << "," <<
                 N << "," <<
                 C << "," <<
                 H << "," <<
                 W << "," <<
                 K << "," <<
                 R << "," <<
                 S << "," <<
                 s << "," <<
                 input_format << "," <<
                 weights_format << "," <<
                 niter << "," <<
                 runtime <<  "," << 
                 (avg / (double)cnt)  <<  std::endl;
}

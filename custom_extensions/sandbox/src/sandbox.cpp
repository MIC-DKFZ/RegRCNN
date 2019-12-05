// ------------------------------------------------------------------
// Faster R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Shaoqing Ren, rewritten in C++ by Gregor Ramien
// ------------------------------------------------------------------
//#include <THC/THC.h>
//#include <TH/TH.h>

#include <torch/extension.h>
#include <iostream>

#include "sandbox.h"

//#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0)) // divide m by n, add +1 if there is a remainder
#define getNBlocks(m,n) ( (m+n-1) / (n) ) // m = nr of total (required) threads, n = nr of threads per block.
int const threadsPerBlock = sizeof(unsigned long long) * 8;

//---- declarations that will be defined in cuda kernel
void add_cuda(int n=1<<3);
void nms_cuda(at::Tensor *boxes, at::Tensor *scores, float thresh);
//-----------------------------------------------------------------

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void sandbox() {

    //std::cout<< "number: "<< number << std::endl;

    torch::Tensor tensor = torch::tensor({0,1,2,3,4,5}, at::kLong).view({2,3});
    std::cout<< "tensor: " << tensor << std::endl;
    std::cout<< "tensor shape: " << at::size(tensor,0) << ", " << at::size(tensor,1) << std::endl;
    return;
}

void add(){
    //tutorial function: add two arrays (x,y) of length n
    add_cuda();
}

//void nms(at::Tensor boxes, at::Tensor scores, float thresh) {
void nms() {

    // x1, y1, x2, y2
    at::Tensor boxes = torch::tensor({
        {20, 10, 60, 40},
        {10, 20, 40, 60},
        {20, 20, 80, 50}
    }, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
    std::cout << "boxes: \n" << boxes << std::endl;
    at::Tensor scores = torch::tensor({
        0.5,
        0.6,
        0.7
    }, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    std::cout<< "scores: \n" << scores << std::endl;

    CHECK_INPUT(boxes); CHECK_INPUT(scores);

    int boxes_num = at::size(boxes,0);
    int boxes_dim = at::size(boxes,1);

    std::cout << "boxes shape: " << boxes_num << ", " << boxes_dim << std::endl;

    float * boxes_dev; unsigned long long * mask_dev; float thresh = 0.05;

    dim3 blocks((boxes_num, threadsPerBlock);
    int threads(threadsPerBlock);


}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sandbox", &sandbox, "Sandy Box Function");
  m.def("nms", &nms, "NMS in cpp");
  m.def("add", &add, "tutorial add function");

}
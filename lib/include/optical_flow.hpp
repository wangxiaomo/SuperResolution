// Copyright (c) 2012, Vladislav Vinogradov (jet47)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the <organization> nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#ifndef __OPENCV_SR_OPTICAL_FLOW_HPP__
#define __OPENCV_SR_OPTICAL_FLOW_HPP__

#include <opencv2/core/core.hpp>
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        class SUPER_RESOLUTION_EXPORT DenseOpticalFlow : public Algorithm
        {
        public:
            virtual void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2 = noArray()) = 0;
            virtual void collectGarbage() = 0;
        };

        SUPER_RESOLUTION_EXPORT Ptr<DenseOpticalFlow> createOptFlowFarneback();
        SUPER_RESOLUTION_EXPORT Ptr<DenseOpticalFlow> createOptFlowSimple();
        SUPER_RESOLUTION_EXPORT Ptr<DenseOpticalFlow> createOptFlowDualTVL1();

        SUPER_RESOLUTION_EXPORT Ptr<DenseOpticalFlow> createOptFlowBrox_GPU();
        SUPER_RESOLUTION_EXPORT Ptr<DenseOpticalFlow> createOptFlowPyrLK_GPU();
        SUPER_RESOLUTION_EXPORT Ptr<DenseOpticalFlow> createOptFlowFarneback_GPU();
        SUPER_RESOLUTION_EXPORT Ptr<DenseOpticalFlow> createOptFlowDualTVL1_GPU();
    }
}

#endif // __OPENCV_SR_OPTICAL_FLOW_HPP__

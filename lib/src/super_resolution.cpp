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

#include "super_resolution.hpp"
#include "optical_flow.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;
using namespace cv::videostab;

bool cv::superres::initModule_superres()
{
    bool all = true;

    all &= !createOptFlowFarneback().empty();
    all &= !createOptFlowSimple().empty();
    all &= !createOptFlowDualTVL1().empty();
    all &= !createOptFlowBrox_GPU().empty();
    all &= !createOptFlowPyrLK_GPU().empty();
    all &= !createOptFlowFarneback_GPU().empty();
    all &= !createOptFlowDualTVL1_GPU().empty();

    all &= !createSuperResBTVL1().empty();
    all &= !createSuperResBTVL1_GPU().empty();

    return all;
}

cv::superres::SuperResolution::SuperResolution()
{
    frameSource = new NullFrameSource();
    firstCall = true;
}

void cv::superres::SuperResolution::setFrameSource(const Ptr<IFrameSource>& frameSource)
{
    this->frameSource = frameSource;
    firstCall = true;
}

void cv::superres::SuperResolution::reset()
{
    this->frameSource->reset();
    firstCall = true;
}

Mat cv::superres::SuperResolution::nextFrame()
{
    if (firstCall)
    {
        initImpl(frameSource);
        firstCall = false;
    }

    return processImpl(frameSource);
}

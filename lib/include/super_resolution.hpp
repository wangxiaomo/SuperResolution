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

#ifndef __SUPER_RESOLUTION_HPP__
#define __SUPER_RESOLUTION_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/videostab/motion_core.hpp>
#include <opencv2/videostab/frame_source.hpp>
#include "super_resolution_export.h"

namespace cv
{
    namespace superres
    {
        using videostab::IFrameSource;
        using videostab::NullFrameSource;
        using videostab::VideoFileSource;

        using cv::videostab::MotionModel;
        using cv::videostab::MM_TRANSLATION;
        using cv::videostab::MM_TRANSLATION_AND_SCALE;
        using cv::videostab::MM_ROTATION;
        using cv::videostab::MM_RIGID;
        using cv::videostab::MM_SIMILARITY;
        using cv::videostab::MM_AFFINE;
        using cv::videostab::MM_HOMOGRAPHY;
        using cv::videostab::MM_UNKNOWN; // General motion via optical flow

        enum BlurModel
        {
            BLUR_BOX,
            BLUR_GAUSS
        };

        enum SRMethod
        {
            SR_BILATERAL_TOTAL_VARIATION,
            SR_TV_L1,
            SR_METHOD_MAX
        };

        class SUPER_RESOLUTION_EXPORT SuperResolution : public Algorithm, public IFrameSource
        {
        public:
            static Ptr<SuperResolution> create(SRMethod method, bool useGpu = false);

            virtual ~SuperResolution();

            void setFrameSource(const Ptr<IFrameSource>& frameSource);

            void reset();
            Mat nextFrame();

        protected:
            SuperResolution();

            virtual void initImpl(Ptr<IFrameSource>& frameSource) = 0;
            virtual Mat processImpl(Ptr<IFrameSource>& frameSource) = 0;

        private:
            Ptr<IFrameSource> frameSource;
            bool firstCall;
        };

        SUPER_RESOLUTION_EXPORT bool initModule_superres();
    }
}

#endif // __SUPER_RESOLUTION_HPP__

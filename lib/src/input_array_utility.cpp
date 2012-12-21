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

#include "input_array_utility.hpp"
#include <limits>
#include <opencv2/core/opengl_interop.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/gpu/gpu.hpp>

using namespace std;
using namespace cv;
using namespace cv::gpu;

Mat getMat(InputArray arr, Mat& buf)
{
    switch (arr.kind())
    {
    case _InputArray::GPU_MAT:
        arr.getGpuMat().download(buf);
        return buf;

    case _InputArray::OPENGL_BUFFER:
        arr.getGlBuffer().copyTo(buf);
        return buf;

    case _InputArray::OPENGL_TEXTURE2D:
        arr.getGlTexture2D().copyTo(buf);
        return buf;

    default:
        return arr.getMat();
    }
}

GpuMat getGpuMat(InputArray arr, GpuMat& buf)
{
    switch (arr.kind())
    {
    case _InputArray::GPU_MAT:
        return arr.getGpuMat();

    case _InputArray::OPENGL_BUFFER:
        arr.getGlBuffer().copyTo(buf);
        return buf;

    case _InputArray::OPENGL_TEXTURE2D:
        arr.getGlTexture2D().copyTo(buf);
        return buf;

    default:
        buf.upload(arr.getMat());
        return buf;
    }
}

namespace
{
    void mat2mat(InputArray src, OutputArray dst)
    {
        src.getMat().copyTo(dst);
    }
    void arr2buf(InputArray src, OutputArray dst)
    {
        dst.getGlBufferRef().copyFrom(src);
    }
    void arr2tex(InputArray src, OutputArray dst)
    {
        dst.getGlTexture2D().copyFrom(src);
    }
    void mat2gpu(InputArray src, OutputArray dst)
    {
        dst.getGpuMatRef().upload(src.getMat());
    }
    void buf2arr(InputArray src, OutputArray dst)
    {
        src.getGlBuffer().copyTo(dst);
    }
    void tex2arr(InputArray src, OutputArray dst)
    {
        src.getGlTexture2D().copyTo(dst);
    }
    void gpu2mat(InputArray src, OutputArray dst)
    {
        GpuMat d = src.getGpuMat();
        dst.create(d.size(), d.type());
        Mat m = dst.getMat();
        d.download(m);
    }
    void gpu2gpu(InputArray src, OutputArray dst)
    {
        src.getGpuMat().copyTo(dst.getGpuMatRef());
    }
}

void copy(InputArray src, OutputArray dst)
{
    typedef void (*func_t)(InputArray src, OutputArray dst);
    static const func_t funcs[10][10] =
    {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu},
        {0, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, mat2mat, arr2buf, arr2tex, mat2gpu},
        {0, buf2arr, buf2arr, buf2arr, buf2arr, buf2arr, buf2arr, buf2arr, buf2arr, buf2arr},
        {0, tex2arr, tex2arr, tex2arr, tex2arr, tex2arr, tex2arr, tex2arr, tex2arr, tex2arr},
        {0, gpu2mat, gpu2mat, gpu2mat, gpu2mat, gpu2mat, gpu2mat, arr2buf, arr2tex, gpu2gpu}
    };

    const int src_kind = src.kind() >> _InputArray::KIND_SHIFT;
    const int dst_kind = dst.kind() >> _InputArray::KIND_SHIFT;

    CV_DbgAssert( src_kind >= 0 && src_kind < 10 );
    CV_DbgAssert( dst_kind >= 0 && dst_kind < 10 );

    const func_t func = funcs[src_kind][dst_kind];
    CV_DbgAssert( func != 0 );

    func(src, dst);
}

namespace
{
    void convertToCn(InputArray src, OutputArray dst, int cn)
    {
        CV_DbgAssert( src.channels() == 1 || src.channels() == 3 || src.channels() == 4 );
        CV_DbgAssert( cn == 1 || cn == 3 || cn == 4 );

        static const int codes[5][5] =
        {
            {-1, -1, -1, -1, -1},
            {-1, -1, -1, COLOR_GRAY2BGR, COLOR_GRAY2BGRA},
            {-1, -1, -1, -1, -1},
            {-1, COLOR_BGR2GRAY, -1, -1, COLOR_BGR2BGRA},
            {-1, COLOR_BGRA2GRAY, -1, COLOR_BGRA2BGR, -1},
        };

        const int code = codes[src.channels()][cn];
        CV_DbgAssert( code >= 0 );

        switch (src.kind())
        {
        case _InputArray::GPU_MAT:
            cvtColor(src.getGpuMat(), dst.getGpuMatRef(), code, cn);

        default:
            cvtColor(src, dst, code, cn);
        }
    }

    void convertToDepth(InputArray src, OutputArray dst, int depth)
    {
        CV_DbgAssert( src.depth() <= CV_64F );
        CV_DbgAssert( depth == CV_8U || depth == CV_32F );

        static const double maxVals[] =
        {
            numeric_limits<uchar>::max(),
            numeric_limits<schar>::max(),
            numeric_limits<ushort>::max(),
            numeric_limits<short>::max(),
            numeric_limits<int>::max(),
            1.0,
            1.0,
        };

        const double scale = maxVals[depth] / maxVals[src.depth()];

        switch (src.kind())
        {
        case _InputArray::GPU_MAT:
            src.getGpuMat().convertTo(dst.getGpuMatRef(), depth, scale);

        default:
            src.getMat().convertTo(dst, depth, scale);
        }
    }
}

Mat convertToType(const Mat& src, int type, Mat& buf0, Mat& buf1)
{
    if (src.type() == type)
        return src;

    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (src.depth() == depth)
    {
        convertToCn(src, buf0, cn);
        return buf0;
    }

    if (src.channels() == cn)
    {
        convertToDepth(src, buf1, depth);
        return buf1;
    }

    convertToCn(src, buf0, cn);
    convertToDepth(buf0, buf1, depth);
    return buf1;
}

GpuMat convertToType(const GpuMat& src, int type, GpuMat& buf0, GpuMat& buf1)
{
    if (src.type() == type)
        return src;

    const int depth = CV_MAT_DEPTH(type);
    const int cn = CV_MAT_CN(type);

    if (src.depth() == depth)
    {
        convertToCn(src, buf0, cn);
        return buf0;
    }

    if (src.channels() == cn)
    {
        convertToDepth(src, buf1, depth);
        return buf1;
    }

    convertToCn(src, buf0, cn);
    convertToDepth(buf0, buf1, depth);
    return buf1;
}

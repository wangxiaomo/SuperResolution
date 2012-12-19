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

#include "optical_flow.hpp"
#include <opencv2/video/tracking.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/core/internal.hpp>
#include "input_array_utility.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::superres;

///////////////////////////////////////////////////////////////////
// CpuOpticalFlow

namespace
{
    class CpuOpticalFlow : public DenseOpticalFlow
    {
    public:
        explicit CpuOpticalFlow(int work_type);

        void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
        void collectGarbage();

    protected:
        virtual void call(const Mat& input0, const Mat& input1, OutputArray dst) = 0;

    private:
        int work_type;
        Mat buf[6];
        Mat flow;
        Mat flows[2];
    };

    CpuOpticalFlow::CpuOpticalFlow(int _work_type) : work_type(_work_type)
    {
    }

    void CpuOpticalFlow::calc(InputArray _frame0, InputArray _frame1, OutputArray _flow1, OutputArray _flow2)
    {
        Mat frame0 = ::getMat(_frame0, buf[0]);
        Mat frame1 = ::getMat(_frame1, buf[1]);

        CV_DbgAssert( frame1.type() == frame0.type() );
        CV_DbgAssert( frame1.size() == frame0.size() );

        Mat input0 = ::convertToType(frame0, work_type, buf[2], buf[3]);
        Mat input1 = ::convertToType(frame1, work_type, buf[4], buf[5]);

        if (!_flow2.needed() && _flow1.kind() != _InputArray::GPU_MAT)
        {
            call(input0, input1, _flow1);
            return;
        }

        call(input0, input1, flow);

        if (!_flow2.needed())
        {
            ::copy(flow, _flow1);
        }
        else
        {
            split(flow, flows);

            ::copy(flows[0], _flow1);
            ::copy(flows[1], _flow2);
        }
    }

    void CpuOpticalFlow::collectGarbage()
    {
        for (int i = 0; i < 6; ++i)
            buf[i].release();
        flow.release();
        flows[0].release();
        flows[1].release();
    }
}

///////////////////////////////////////////////////////////////////
// Farneback

namespace
{
    class Farneback : public CpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        Farneback();

    protected:
        void call(const Mat& input0, const Mat& input1, OutputArray dst);

    private:
        double pyrScale;
        int numLevels;
        int winSize;
        int numIters;
        int polyN;
        double polySigma;
        int flags;
    };

    CV_INIT_ALGORITHM(Farneback, "DenseOpticalFlow.Farneback",
                      obj.info()->addParam(obj, "pyrScale", obj.pyrScale);
                      obj.info()->addParam(obj, "numLevels", obj.numLevels);
                      obj.info()->addParam(obj, "winSize", obj.winSize);
                      obj.info()->addParam(obj, "numIters", obj.numIters);
                      obj.info()->addParam(obj, "polyN", obj.polyN);
                      obj.info()->addParam(obj, "polySigma", obj.polySigma);
                      obj.info()->addParam(obj, "flags", obj.flags));

    Farneback::Farneback() : CpuOpticalFlow(CV_8UC1)
    {
        pyrScale = 0.5;
        numLevels = 5;
        winSize = 13;
        numIters = 10;
        polyN = 5;
        polySigma = 1.1;
        flags = 0;
    }

    void Farneback::call(const Mat& input0, const Mat& input1, OutputArray dst)
    {
        calcOpticalFlowFarneback(input0, input1, dst, pyrScale, numLevels, winSize, numIters, polyN, polySigma, flags);
    }
}

Ptr<DenseOpticalFlow> cv::superres::createOptFlowFarneback()
{
    return new Farneback;
}

///////////////////////////////////////////////////////////////////
// Simple

namespace
{
    class Simple : public CpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        Simple();

    protected:
        void call(const Mat& input0, const Mat& input1, OutputArray dst);

    private:
        int layers;
        int averagingBlockSize;
        int maxFlow;
        double sigmaDist;
        double sigmaColor;
        int postProcessWindow;
        double sigmaDistFix;
        double sigmaColorFix;
        double occThr;
        int upscaleAveragingRadius;
        double upscaleSigmaDist;
        double upscaleSigmaColor;
        double speedUpThr;
    };

    CV_INIT_ALGORITHM(Simple, "DenseOpticalFlow.Simple",
                      obj.info()->addParam(obj, "layers", obj.layers);
                      obj.info()->addParam(obj, "averagingBlockSize", obj.averagingBlockSize);
                      obj.info()->addParam(obj, "maxFlow", obj.maxFlow);
                      obj.info()->addParam(obj, "sigmaDist", obj.sigmaDist);
                      obj.info()->addParam(obj, "sigmaColor", obj.sigmaColor);
                      obj.info()->addParam(obj, "postProcessWindow", obj.postProcessWindow);
                      obj.info()->addParam(obj, "sigmaDistFix", obj.sigmaDistFix);
                      obj.info()->addParam(obj, "sigmaColorFix", obj.sigmaColorFix);
                      obj.info()->addParam(obj, "occThr", obj.occThr);
                      obj.info()->addParam(obj, "upscaleAveragingRadius", obj.upscaleAveragingRadius);
                      obj.info()->addParam(obj, "upscaleSigmaDist", obj.upscaleSigmaDist);
                      obj.info()->addParam(obj, "upscaleSigmaColor", obj.upscaleSigmaColor);
                      obj.info()->addParam(obj, "speedUpThr", obj.speedUpThr));

    Simple::Simple() : CpuOpticalFlow(CV_8UC3)
    {
        layers = 3;
        averagingBlockSize = 2;
        maxFlow = 4;
        sigmaDist = 4.1;
        sigmaColor = 25.5;
        postProcessWindow = 18;
        sigmaDistFix = 55.0;
        sigmaColorFix = 25.5;
        occThr = 0.35;
        upscaleAveragingRadius = 18;
        upscaleSigmaDist = 55.0;
        upscaleSigmaColor = 25.5;
        speedUpThr = 10;
    }

    void Simple::call(const Mat& input0, const Mat& input1, OutputArray dst)
    {
        calcOpticalFlowSF(input0, input1, dst,
                          layers,
                          averagingBlockSize,
                          maxFlow,
                          sigmaDist,
                          sigmaColor,
                          postProcessWindow,
                          sigmaDistFix,
                          sigmaColorFix,
                          occThr,
                          upscaleAveragingRadius,
                          upscaleSigmaDist,
                          upscaleSigmaColor,
                          speedUpThr);
    }
}

Ptr<DenseOpticalFlow> cv::superres::createOptFlowSimple()
{
    return new Simple;
}

///////////////////////////////////////////////////////////////////
// DualTVL1

namespace
{
    class DualTVL1 : public CpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        DualTVL1();

        void collectGarbage();

    protected:
        void call(const Mat& input0, const Mat& input1, OutputArray dst);

    private:
        double tau;
        double lambda;
        double theta;
        int    nscales;
        int    warps;
        double epsilon;
        int iterations;
        bool useInitialFlow;

        OpticalFlowDual_TVL1 alg;
    };

    CV_INIT_ALGORITHM(DualTVL1, "DenseOpticalFlow.DualTVL1",
                      obj.info()->addParam(obj, "tau", obj.tau);
                      obj.info()->addParam(obj, "lambda", obj.lambda);
                      obj.info()->addParam(obj, "theta", obj.theta);
                      obj.info()->addParam(obj, "nscales", obj.nscales);
                      obj.info()->addParam(obj, "warps", obj.warps);
                      obj.info()->addParam(obj, "epsilon", obj.epsilon);
                      obj.info()->addParam(obj, "iterations", obj.iterations);
                      obj.info()->addParam(obj, "useInitialFlow", obj.useInitialFlow));

    DualTVL1::DualTVL1() : CpuOpticalFlow(CV_8UC1)
    {
        tau = alg.tau;
        lambda = alg.lambda;
        theta = alg.theta;
        nscales = alg.nscales;
        warps = alg.warps;
        epsilon = alg.epsilon;
        iterations = alg.iterations;
        useInitialFlow = alg.useInitialFlow;
    }

    void DualTVL1::call(const Mat& input0, const Mat& input1, OutputArray dst)
    {
        alg.tau = tau;
        alg.lambda = lambda;
        alg.theta = theta;
        alg.nscales = nscales;
        alg.warps = warps;
        alg.epsilon = epsilon;
        alg.iterations = iterations;
        alg.useInitialFlow = useInitialFlow;

        alg(input0, input1, dst);
    }

    void DualTVL1::collectGarbage()
    {
        alg.collectGarbage();
        CpuOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlow> cv::superres::createOptFlowDualTVL1()
{
    return new DualTVL1;
}

///////////////////////////////////////////////////////////////////
// GpuOpticalFlow

namespace
{
    class GpuOpticalFlow : public DenseOpticalFlow
    {
    public:
        explicit GpuOpticalFlow(int work_type);

        void calc(InputArray frame0, InputArray frame1, OutputArray flow1, OutputArray flow2);
        void collectGarbage();

    protected:
        virtual void call(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2) = 0;

    private:
        int work_type;
        GpuMat buf[6];
        GpuMat u, v, flow;
    };

    GpuOpticalFlow::GpuOpticalFlow(int _work_type) : work_type(_work_type)
    {
    }

    void GpuOpticalFlow::calc(InputArray _frame0, InputArray _frame1, OutputArray _flow1, OutputArray _flow2)
    {
        GpuMat frame0 = ::getGpuMat(_frame0, buf[0]);
        GpuMat frame1 = ::getGpuMat(_frame1, buf[1]);

        CV_DbgAssert( frame1.type() == frame0.type() );
        CV_DbgAssert( frame1.size() == frame0.size() );

        GpuMat input0 = ::convertToType(frame0, work_type, buf[2], buf[3]);
        GpuMat input1 = ::convertToType(frame1, work_type, buf[4], buf[5]);

        if (_flow2.needed() && _flow1.kind() == _InputArray::GPU_MAT && _flow2.kind() == _InputArray::GPU_MAT)
        {
            call(input0, input1, _flow1.getGpuMatRef(), _flow2.getGpuMatRef());
            return;
        }

        call(input0, input1, u, v);

        if (_flow2.needed())
        {
            ::copy(u, _flow1);
            ::copy(v, _flow2);
        }
        else
        {
            GpuMat src[] = {u, v};
            merge(src, 2, flow);
            ::copy(flow, _flow1);
        }
    }

    void GpuOpticalFlow::collectGarbage()
    {
        for (int i = 0; i < 6; ++i)
            buf[i].release();
        u.release();
        v.release();
        flow.release();
    }
}

///////////////////////////////////////////////////////////////////
// Brox_GPU

namespace
{
    class Brox_GPU : public GpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        Brox_GPU();

        void collectGarbage();

    protected:
        void call(const gpu::GpuMat& input0, const gpu::GpuMat& input1, gpu::GpuMat& dst1, gpu::GpuMat& dst2);

    private:
        double alpha;
        double gamma;
        double scaleFactor;
        int innerIterations;
        int outerIterations;
        int solverIterations;

        BroxOpticalFlow alg;
    };

    CV_INIT_ALGORITHM(Brox_GPU, "DenseOpticalFlow.Brox_GPU",
                      obj.info()->addParam(obj, "alpha", obj.alpha, false, 0, 0, "Flow smoothness");
                      obj.info()->addParam(obj, "gamma", obj.gamma, false, 0, 0, "Gradient constancy importance");
                      obj.info()->addParam(obj, "scaleFactor", obj.scaleFactor, false, 0, 0, "Pyramid scale factor");
                      obj.info()->addParam(obj, "innerIterations", obj.innerIterations, false, 0, 0, "Number of lagged non-linearity iterations (inner loop)");
                      obj.info()->addParam(obj, "outerIterations", obj.outerIterations, false, 0, 0, "Number of warping iterations (number of pyramid levels)");
                      obj.info()->addParam(obj, "solverIterations", obj.solverIterations, false, 0, 0, "Number of linear system solver iterations"));

    Brox_GPU::Brox_GPU() : GpuOpticalFlow(CV_32FC1), alg(0.197, 50.0, 0.8, 10, 77, 10)
    {
        alpha = alg.alpha;
        gamma = alg.gamma;
        scaleFactor = alg.scale_factor;
        innerIterations = alg.inner_iterations;
        outerIterations = alg.outer_iterations;
        solverIterations = alg.solver_iterations;
    }

    void Brox_GPU::call(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
    {
        alg.alpha = alpha;
        alg.gamma = gamma;
        alg.scale_factor = scaleFactor;
        alg.inner_iterations = innerIterations;
        alg.outer_iterations = outerIterations;
        alg.solver_iterations = solverIterations;

        alg(input0, input1, dst1, dst2);
    }

    void Brox_GPU::collectGarbage()
    {
        alg.buf.release();
        GpuOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlow> cv::superres::createOptFlowBrox_GPU()
{
    return new Brox_GPU;
}

///////////////////////////////////////////////////////////////////
// PyrLK_GPU

namespace
{
    class PyrLK_GPU : public GpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        PyrLK_GPU();

        void collectGarbage();

    protected:
        void call(const gpu::GpuMat& input0, const gpu::GpuMat& input1, gpu::GpuMat& dst1, gpu::GpuMat& dst2);

    private:
        int winSize;
        int maxLevel;
        int iterations;

        PyrLKOpticalFlow alg;
    };

    CV_INIT_ALGORITHM(PyrLK_GPU, "DenseOpticalFlow.PyrLK_GPU",
                      obj.info()->addParam(obj, "winSize", obj.winSize);
                      obj.info()->addParam(obj, "maxLevel", obj.maxLevel);
                      obj.info()->addParam(obj, "iterations", obj.iterations));

    PyrLK_GPU::PyrLK_GPU() : GpuOpticalFlow(CV_8UC1)
    {
        winSize = alg.winSize.width;
        maxLevel = alg.maxLevel;
        iterations = alg.iters;
    }

    void PyrLK_GPU::call(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
    {
        alg.winSize.width = winSize;
        alg.winSize.height = winSize;
        alg.maxLevel = maxLevel;
        alg.iters = iterations;

        alg.dense(input0, input1, dst1, dst2);
    }

    void PyrLK_GPU::collectGarbage()
    {
        alg.releaseMemory();
        GpuOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlow> cv::superres::createOptFlowPyrLK_GPU()
{
    return new PyrLK_GPU;
}

///////////////////////////////////////////////////////////////////
// Farneback_GPU

namespace
{
    class Farneback_GPU : public GpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        Farneback_GPU();

        void collectGarbage();

    protected:
        void call(const gpu::GpuMat& input0, const gpu::GpuMat& input1, gpu::GpuMat& dst1, gpu::GpuMat& dst2);

    private:
        double pyrScale;
        int numLevels;
        int winSize;
        int numIters;
        int polyN;
        double polySigma;
        int flags;

        FarnebackOpticalFlow alg;
    };

    CV_INIT_ALGORITHM(Farneback_GPU, "DenseOpticalFlow.Farneback_GPU",
                      obj.info()->addParam(obj, "pyrScale", obj.pyrScale);
                      obj.info()->addParam(obj, "numLevels", obj.numLevels);
                      obj.info()->addParam(obj, "winSize", obj.winSize);
                      obj.info()->addParam(obj, "numIters", obj.numIters);
                      obj.info()->addParam(obj, "polyN", obj.polyN);
                      obj.info()->addParam(obj, "polySigma", obj.polySigma);
                      obj.info()->addParam(obj, "flags", obj.flags));

    Farneback_GPU::Farneback_GPU() : GpuOpticalFlow(CV_8UC1)
    {
        pyrScale = alg.pyrScale;
        numLevels = alg.numLevels;
        winSize = alg.winSize;
        numIters = alg.numIters;
        polyN = alg.polyN;
        polySigma = alg.polySigma;
        flags = alg.flags;
    }

    void Farneback_GPU::call(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
    {
        alg.pyrScale = pyrScale;
        alg.numLevels = numLevels;
        alg.winSize = winSize;
        alg.numIters = numIters;
        alg.polyN = polyN;
        alg.polySigma = polySigma;
        alg.flags = flags;

        alg(input0, input1, dst1, dst2);
    }

    void Farneback_GPU::collectGarbage()
    {
        alg.releaseMemory();
        GpuOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlow> cv::superres::createOptFlowFarneback_GPU()
{
    return new Farneback_GPU;
}

///////////////////////////////////////////////////////////////////
// DualTVL1_GPU

namespace
{
    class DualTVL1_GPU : public GpuOpticalFlow
    {
    public:
        AlgorithmInfo* info() const;

        DualTVL1_GPU();

        void collectGarbage();

    protected:
        void call(const gpu::GpuMat& input0, const gpu::GpuMat& input1, gpu::GpuMat& dst1, gpu::GpuMat& dst2);

    private:
        double tau;
        double lambda;
        double theta;
        int    nscales;
        int    warps;
        double epsilon;
        int iterations;
        bool useInitialFlow;

        OpticalFlowDual_TVL1_GPU alg;
    };

    CV_INIT_ALGORITHM(DualTVL1_GPU, "DenseOpticalFlow.DualTVL1_GPU",
                      obj.info()->addParam(obj, "tau", obj.tau);
                      obj.info()->addParam(obj, "lambda", obj.lambda);
                      obj.info()->addParam(obj, "theta", obj.theta);
                      obj.info()->addParam(obj, "nscales", obj.nscales);
                      obj.info()->addParam(obj, "warps", obj.warps);
                      obj.info()->addParam(obj, "epsilon", obj.epsilon);
                      obj.info()->addParam(obj, "iterations", obj.iterations);
                      obj.info()->addParam(obj, "useInitialFlow", obj.useInitialFlow));

    DualTVL1_GPU::DualTVL1_GPU() : GpuOpticalFlow(CV_8UC1)
    {
        tau = alg.tau;
        lambda = alg.lambda;
        theta = alg.theta;
        nscales = alg.nscales;
        warps = alg.warps;
        epsilon = alg.epsilon;
        iterations = alg.iterations;
        useInitialFlow = alg.useInitialFlow;
    }

    void DualTVL1_GPU::call(const GpuMat& input0, const GpuMat& input1, GpuMat& dst1, GpuMat& dst2)
    {
        alg.tau = tau;
        alg.lambda = lambda;
        alg.theta = theta;
        alg.nscales = nscales;
        alg.warps = warps;
        alg.epsilon = epsilon;
        alg.iterations = iterations;
        alg.useInitialFlow = useInitialFlow;

        alg(input0, input1, dst1, dst2);
    }

    void DualTVL1_GPU::collectGarbage()
    {
        alg.collectGarbage();
        GpuOpticalFlow::collectGarbage();
    }
}

Ptr<DenseOpticalFlow> cv::superres::createOptFlowDualTVL1_GPU()
{
    return new DualTVL1_GPU;
}

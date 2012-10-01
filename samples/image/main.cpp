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

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include "image_super_resolution.hpp"

using namespace std;
using namespace cv;
using namespace cv::superres;

#define MEASURE_TIME(op, msg) \
    { \
        TickMeter tm; \
        tm.start(); \
        op; \
        tm.stop(); \
        cout << msg << " Time : " << tm.getTimeSec() << " sec" << endl; \
    }

namespace
{
    void addGaussNoise(Mat_<Vec3b>& image, double sigma)
    {
        Mat_<float> noise(image.size());

        vector<Mat> channels;
        split(image, channels);

        Mat_<float> src_f;
        Mat_<float> temp;

        for(int c = 0; c < image.channels(); ++c)
        {
            channels[c].convertTo(src_f, src_f.depth());

            randn(noise, Scalar(0.0), Scalar(sigma));

            add(src_f, noise, temp);

            temp.convertTo(channels[c], channels[c].depth());
        }

        merge(channels, image);
    }

    void addSpikeNoise(Mat_<Vec3b>& image, int val)
    {
        for (int y = 0; y < image.rows; ++y)
        {
            Vec3b* imageRow = image[y];

            for (int x = 0; x < image.cols; ++x)
            {
                if (theRNG().uniform(0 ,val) < 1)
                    imageRow[x] = Vec3b(255, 255, 255);
            }
        }
    }

    Mat createDegradedImage(const Mat& src, Point2d move, int scale)
    {
        const double iscale = 1.0 / scale;

        Mat_<float> M(2, 3);
        M << 1, 0, move.x,
             0, 1, move.y;

        Mat_<Vec3b> shifted;
        warpAffine(src, shifted, M, src.size());

        Mat_<Vec3b> blurred;
        blur(shifted, blurred, Size(scale, scale));

        Mat_<Vec3b> deg;
        resize(blurred, deg, Size(), iscale, iscale, INTER_NEAREST);

        addGaussNoise(deg, 10.0);
        addSpikeNoise(deg, 500);

        return deg;
    }
}

int main(int argc, const char* argv[])
{
    CommandLineParser cmd(argc, argv,
        "{ image i | boy.png | Input image }"
        "{ scale s | 2       | Scale factor }"
        "{ help h  |         | Print help message }"
    );

    if (cmd.has("help"))
    {
        cmd.about("This sample demonstrates Super Resolution algorithms for single image");
        cmd.printMessage();
        return 0;
    }

    const string imageFileName = cmd.get<string>("image");
    const int scale = cmd.get<int>("scale");

    Mat gold = imread(imageFileName);
    if (gold.empty())
    {
        cerr << "Can't open image " << imageFileName << endl;
        return -1;
    }

    Mat src = createDegradedImage(gold, Point2d(0,0), scale);

    // number of input images for super resolution
    const int degImagesCount = 16;
    vector<Mat> degImages(degImagesCount);
    for (int i = 0; i < degImagesCount; ++i)
    {
        Point2d move;
        move.x = theRNG().uniform(0.0, (double)scale);
        move.y = theRNG().uniform(0.0, (double)scale);

        degImages[i] = createDegradedImage(gold, move, scale);
    }

    Ptr<ImageSuperResolution> superRes = ImageSuperResolution::create(IMAGE_SR_BILATERAL_TOTAL_VARIATION);

    superRes->set("scale", scale);

    superRes->train(degImages);

    Mat highResImage;
    MEASURE_TIME(superRes->process(src, highResImage), "Process");

    Mat bicubic;
    resize(src, bicubic, Size(), scale, scale, INTER_CUBIC);

    namedWindow("gold", WINDOW_NORMAL);
    imshow("gold", gold);

    namedWindow("Super Resolution", WINDOW_NORMAL);
    imshow("Super Resolution", highResImage);

    namedWindow("Bi-Cubic Interpolation", WINDOW_NORMAL);
    imshow("Bi-Cubic Interpolation", bicubic);

    waitKey();

    return 0;
}

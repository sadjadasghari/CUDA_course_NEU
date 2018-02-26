bplist00—_WebMainResource’	
^WebResourceURL_WebResourceFrameName_WebResourceData_WebResourceMIMEType_WebResourceTextEncodingName_àhttp://www.cse.uaa.alaska.edu/~ssiewert/a490dmis_code/CUDA/cuda_work/samples/7_CUDALibraries/histEqualizationNPP/histEqualizationNPP.cppPO'ú<html><head><style>.pkt_added {text-decoration:none !important;}</style><style type="text/css"></style></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include &lt;windows.h&gt;
#endif

#include &lt;ImagesCPU.h&gt;
#include &lt;ImagesNPP.h&gt;
#include &lt;ImageIO.h&gt;
#include &lt;Exceptions.h&gt;
#include &lt;string.h&gt;

#include &lt;string&gt;
#include &lt;fstream&gt;
#include &lt;iostream&gt;

#include &lt;npp.h&gt;

#include &lt;helper_cuda.h&gt;

#ifdef WIN32
#define STRCASECMP  _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP  strcasecmp
#define STRNCASECMP strncasecmp
#endif

bool g_bQATest = false;
int  g_nDevice = -1;

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&amp;deviceCount));

    if (deviceCount == 0)
    {
        std::cerr &lt;&lt; "CUDA error: no devices supporting CUDA." &lt;&lt; std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&amp;deviceProp, dev);
    std::cerr &lt;&lt; "cudaSetDevice GPU" &lt;&lt; dev &lt;&lt; " = " &lt;&lt; deviceProp.name &lt;&lt; std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

void parseCommandLineArguments(int argc, char *argv[])
{
    if (argc &gt;= 2)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "qatest") ||
            checkCmdLineFlag(argc, (const char **)argv, "noprompt"))
        {
            g_bQATest = true;
        }
    }
}

void printfNPPinfo(int argc, char *argv[])
{
    const char *sComputeCap[] =
    {
        "No CUDA Capable Device Found",
        "Compute 1.0", "Compute 1.1", "Compute 1.2",  "Compute 1.3",
        "Compute 2.0", "Compute 2.1", "Compute 3.0", NULL
    };

    const NppLibraryVersion *libVer   = nppGetLibVersion();
    NppGpuComputeCapability computeCap = nppGetGpuComputeCapability();

    printf("NPP Library Version %d.%d.%d\n", libVer-&gt;major, libVer-&gt;minor, libVer-&gt;build);

    if (computeCap != 0 &amp;&amp; g_nDevice == -1)
    {
        printf("%s using GPU &lt;%s&gt; with %d SM(s) with", argv[0], nppGetGpuName(), nppGetGpuNumSMs());

        if (computeCap &gt; 0)
        {
            printf(" %s\n", sComputeCap[computeCap]);
        }
        else
        {
            printf(" Unknown Compute Capabilities\n");
        }
    }
    else
    {
        printf("%s\n", sComputeCap[computeCap]);
    }
}

int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        std::string sFilename;
        char *filePath = sdkFindFilePath("Lena.pgm", argv[0]);

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            printf("Error unable to find Lena.pgm\n");
            exit(EXIT_FAILURE);
        }

        // Parse the command line arguments for proper configuration
        parseCommandLineArguments(argc, argv);

        cudaDeviceInit(argc, (const char **)argv);

        printfNPPinfo(argc, argv);

        if (g_bQATest == false &amp;&amp; (g_nDevice == -1) &amp;&amp; argc &gt; 1)
        {
            sFilename = argv[1];
        }

        // if we specify the filename at the command line, then we only test sFilename.
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout &lt;&lt; "histEqualizationNPP opened: &lt;" &lt;&lt; sFilename.data() &lt;&lt; "&gt; successfully!" &lt;&lt; std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout &lt;&lt; "histEqualizationNPP unable to open: &lt;" &lt;&lt; sFilename.data() &lt;&lt; "&gt;" &lt;&lt; std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors &gt; 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string dstFileName = sFilename;

        std::string::size_type dot = dstFileName.rfind('.');

        if (dot != std::string::npos)
        {
            dstFileName = dstFileName.substr(0, dot);
        }

        dstFileName += "_histEqualization.pgm";

        if (argc &gt;= 3 &amp;&amp; !g_bQATest)
        {
            dstFileName = argv[2];
        }

        npp::ImageCPU_8u_C1 oHostSrc;
        npp::loadImage(sFilename, oHostSrc);
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        //
        // allocate arrays for histogram and levels
        //

        const int binCount = 256;
        const int levelCount = binCount + 1; // levels array has one more element

        Npp32s *histDevice = 0;
        Npp32s *levelsDevice = 0;

        NPP_CHECK_CUDA(cudaMalloc((void **)&amp;histDevice,   binCount   * sizeof(Npp32s)));
        NPP_CHECK_CUDA(cudaMalloc((void **)&amp;levelsDevice, levelCount * sizeof(Npp32s)));

        //
        // compute histogram
        //

        NppiSize oSizeROI = {oDeviceSrc.width(), oDeviceSrc.height()}; // full image
        // create device scratch buffer for nppiHistogram
        int nDeviceBufferSize;
        nppiHistogramEvenGetBufferSize_8u_C1R(oSizeROI, levelCount ,&amp;nDeviceBufferSize);
        Npp8u *pDeviceBuffer;
        NPP_CHECK_CUDA(cudaMalloc((void **)&amp;pDeviceBuffer, nDeviceBufferSize));

        // compute levels values on host
        Npp32s levelsHost[levelCount];
        NPP_CHECK_NPP(nppiEvenLevelsHost_32s(levelsHost, levelCount, 0, binCount));
        // compute the histogram
        NPP_CHECK_NPP(nppiHistogramEven_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oSizeROI,
                                               histDevice, levelCount, 0, binCount,
                                               pDeviceBuffer));
        // copy histogram and levels to host memory
        Npp32s histHost[binCount];
        NPP_CHECK_CUDA(cudaMemcpy(histHost, histDevice, binCount * sizeof(Npp32s), cudaMemcpyDeviceToHost));

        Npp32s  lutHost[binCount + 1];

        // fill LUT
        {
            Npp32s *pHostHistogram = histHost;
            Npp32s totalSum = 0;

            for (; pHostHistogram &lt; histHost + binCount; ++pHostHistogram)
            {
                totalSum += *pHostHistogram;
            }

            NPP_ASSERT(totalSum == oSizeROI.width * oSizeROI.height);

            if (totalSum == 0)
            {
                totalSum = 1;
            }

            float multiplier = 1.0f / float(totalSum) * 0xFF;

            Npp32s runningSum = 0;
            Npp32s *pLookupTable = lutHost;

            for (pHostHistogram = histHost; pHostHistogram &lt; histHost + binCount; ++pHostHistogram)
            {
                *pLookupTable = (Npp32s)(runningSum * multiplier + 0.5f);
                pLookupTable++;
                runningSum += *pHostHistogram;
            }

            lutHost[binCount] = 0xFF; // last element is always 1
        }

        //
        // apply LUT transformation to the image
        //
        // Create a device image for the result.
        npp::ImageNPP_8u_C1 oDeviceDst(oDeviceSrc.size());

#if CUDART_VERSION &gt;= 5000
        // Note for CUDA 5.0, that nppiLUT_Linear_8u_C1R requires these pointers to be in GPU device memory
        Npp32s  *lutDevice  = 0;
        Npp32s  *lvlsDevice = 0;

        NPP_CHECK_CUDA(cudaMalloc((void **)&amp;lutDevice,  sizeof(Npp32s) * (binCount + 1)));
        NPP_CHECK_CUDA(cudaMalloc((void **)&amp;lvlsDevice, sizeof(Npp32s) * binCount));

        NPP_CHECK_CUDA(cudaMemcpy(lutDevice , lutHost,   sizeof(Npp32s) * (binCount+1), cudaMemcpyHostToDevice));
        NPP_CHECK_CUDA(cudaMemcpy(lvlsDevice, levelsHost, sizeof(Npp32s) * binCount   , cudaMemcpyHostToDevice));

        NPP_CHECK_NPP(nppiLUT_Linear_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                            oDeviceDst.data(), oDeviceDst.pitch(),
                                            oSizeROI,
                                            lutDevice, // value and level arrays are in GPU device memory
                                            lvlsDevice,
                                            binCount+1));

        NPP_CHECK_CUDA(cudaFree(lutDevice));
        NPP_CHECK_CUDA(cudaFree(lvlsDevice));
#else
        NPP_CHECK_NPP(nppiLUT_Linear_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
                                            oDeviceDst.data(), oDeviceDst.pitch(),
                                            oSizeROI,
                                            lutHost, // value and level arrays are in host memory
                                            levelsHost,
                                            binCount+1));
#endif

        // copy the result image back into the storage that contained the
        // input image
        npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

        // save the result
        npp::saveImage(dstFileName.c_str(), oHostDst);

        std::cout &lt;&lt; "Saved image file " &lt;&lt; dstFileName &lt;&lt; std::endl;
        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &amp;rException)
    {
        std::cerr &lt;&lt; "Program error! The following exception occurred: \n";
        std::cerr &lt;&lt; rException &lt;&lt; std::endl;
        std::cerr &lt;&lt; "Aborting." &lt;&lt; std::endl;
        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr &lt;&lt; "Program error! An unknow type of exception occurred. \n";
        std::cerr &lt;&lt; "Aborting." &lt;&lt; std::endl;
        exit(EXIT_FAILURE);
    }

    return 0;
}

</pre></body></html>Ztext/plainUUTF-8    ( 7 N ` v î (¿(À                           (—
/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <fstream>
#include <string>


#include "cudaNormalize.h"
#include "cudaResize.h"
#include "cudaFont.h"
#include "imageNet.h"


#define DEFAULT_CAMERA -1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)	
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>


#define LOW_BYTE (1 << 10) //1kb
//GOOD
#define UP_BYTE (1 << 23) //8MB
//BAD
//#define UP_BYTE (1 << 29) //8MB
#define MAX UP_BYTE / sizeof(double)

double a[MAX] = {1};

double current_time(void)
{
    double timestamp;

    struct timeval tv;
    gettimeofday(&tv, 0);

    timestamp = (double)((double)(tv.tv_sec * 1e6) + (double)tv.tv_usec);
    return timestamp;
}

void test_band_width(int size)
{
    int i;
    volatile double r = 0;
    for (i = 0; i < size; i += 16)
    {
        r += a[i];
    }
}

void cpu_bandwithrate(int up , int low)
{
    int k, size, n = 0;
    double cycles;
    double t_start = 0.0, t_end = 0.0, time = 0.0;

    int pos = 0;
    for (k = 1<<up; k >= 1<<low; k >>= 1)
    {
        size = k / sizeof(double);
        t_start = current_time();
        test_band_width(size);
        t_end = current_time();
        time = (t_end - t_start);
        pos++;
    }
    
}	
		
		
bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


void not_so_cool_code(bool &haveDisturbance,bool &blackScreen)
{
	int up = 10;
	int low =6;
	if(haveDisturbance)
		up = 23;
	   
	cpu_bandwithrate(up ,low);
}

void reload_system_caps(bool &haveDisturbance,bool &blackScreen)
{
	haveDisturbance = false;
	blackScreen = false;			
	
	// read from file
	std::fstream file;
	file.open("input_state.txt", std::ios::in);
	if (file.is_open())
	{
		std::string str; 
		std::getline(file, str);
		if (str == "DISTURBANCE") haveDisturbance = true;
		if (str == "BLACKSCREEN") blackScreen = true;
	}
}

int main( int argc, char** argv )
{
	printf("imagenet-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	

	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);
	
	if( !camera )
	{
		printf("\nimagenet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\nimagenet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create imageNet
	 */
	imageNet* net = imageNet::Create(argc, argv);
	
	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\nimagenet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("imagenet-camera:  failed to create openGL texture\n");
	}
	
	
	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\nimagenet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\nimagenet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	bool haveDisturbance = false;
	bool blackScreen = false;

	time_t start = time (NULL);
	srand (time(NULL));

	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;

		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\nimagenet-camera:  failed to capture frame\n");
		//else
		//	printf("imagenet-camera:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);
		
		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
		if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
			printf("imagenet-camera:  failed to convert from NV12 to RGBA\n");


		// classify image
		const int img_class = net->Classify((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), &confidence);
	
		if( img_class >= 0 )
		{
			printf("imagenet-camera:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net->GetClassDesc(img_class));	

			if( font != NULL )
			{
				char str[256];
				sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
	
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 0, 0, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}
			
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT %i.%i.%i | %s | %s | %04.1f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->GetNetworkName(), precisionTypeToStr(net->GetPrecision()), display->GetFPS());
				display->SetTitle(str);	
			}	
		}	


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				if (time(NULL) - start >= 5)
				{
					start = time(NULL);
					reload_system_caps(haveDisturbance,blackScreen);
				}

				if (haveDisturbance)
				{
					if ((rand() % 100) < 20)
					{
						// rescale image pixel intensities for display
						CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
										(float4*)imgRGBA, make_float2(0.0f, 1.0f), 
										camera->GetWidth()/4, camera->GetHeight()/4));
					}
					else
					{
						// rescale image pixel intensities for display
						CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
										(float4*)imgRGBA, make_float2(0.0f, 1.0f), 
										camera->GetWidth(), camera->GetHeight()));						
					}
				}
				else if (blackScreen)
				{
						// rescale image pixel intensities for display
						CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
										(float4*)imgRGBA, make_float2(0.0f, 0.0f), 
										camera->GetWidth(), camera->GetHeight()));
				}
				else
				{
						// rescale image pixel intensities for display
						CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
										(float4*)imgRGBA, make_float2(0.0f, 1.0f), 
										camera->GetWidth(), camera->GetHeight()));
				}

				not_so_cool_code(haveDisturbance,blackScreen);



				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				texture->Render(100,100);
			}

			display->EndRender();
		}
	}
	
	printf("\nimagenet-camera:  un-initializing video device\n");
	
	
	/*
	 * shutdown the camera device
	 */
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	printf("imagenet-camera:  video device has been un-initialized.\n");
	printf("imagenet-camera:  this concludes the test of the video device.\n");
	return 0;
}


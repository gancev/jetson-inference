/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "detectNet.h"
#include "loadImage.h"

#include "cudaMappedMemory.h"


// main entry point
int main( int argc, char** argv )
{
	printf("detectnet-console\n  args (%i):  ", argc);
	
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	
	// retrieve filename argument
	if( argc < 2 )
	{
		printf("detectnet-console:   input image filename required\n");
		return 0;
	}
	
	const char* imgFilename = argv[1];
	

	// create detectNet
	//detectNet* net = detectNet::Create("ped-100/deploy.prototxt", "ped-100/snapshot_iter_70800.caffemodel", "ped-100/mean.binaryproto" );
	//detectNet* net = detectNet::Create("multiped-90/deploy.prototxt", "multiped-90/snapshot_iter_32040.caffemodel", "multiped-90/mean.binaryproto" );
	detectNet* net = detectNet::Create("multiped-500/deploy.prototxt", "multiped-500/snapshot_iter_178000.caffemodel", "multiped-500/mean.binaryproto" );
	
	if( !net )
	{
		printf("detectnet-console:   failed to initialize detectNet\n");
		return 0;
	}
	
	// alloc memory for bounding box & confidence value output arrays
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();
	
	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;
	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-console:  failed to alloc output memory\n");
		return 0;
	}
	
	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}
	
	// classify image
	int numBoundingBoxes = maxBoxes;
	
	if( !net->Detect(imgCUDA, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU) )
		printf("detectnet-console:  failed to classify '%s'\n", imgFilename);
	else if( argc > 2 )		// if the user supplied an output filename
	{
		// perform output bounding box overlay
		printf("%i bounding boxes detected\n", numBoundingBoxes);
		
		int lastClass = 0;
		int lastStart = 0;
		
		for( int n=0; n < numBoundingBoxes; n++ )
		{
			const int nc = confCPU[n*2+1];
			float* bb = bbCPU + (n * 4);
			
			printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
			
			if( nc != lastClass || n == (numBoundingBoxes - 1) )
			{
				if( !net->DrawBoxes(imgCUDA, imgCUDA, imgWidth, imgHeight, bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
					printf("detectnet-console:  failed to draw boxes\n");
					
				lastClass = nc;
				lastStart = n;
			}
		}
		
		CUDA(cudaThreadSynchronize());
		
		// save image to disk
		printf("detectnet-console:  writing %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);
		
		if( !saveImageRGBA(argv[2], (float4*)imgCPU, imgWidth, imgHeight, 255.0f) )
			printf("detectnet-console:  failed saving %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);
		else	
			printf("detectnet-console:  successfully wrote %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);
		
	}
	//printf("detectnet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, "pedestrian");
	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}
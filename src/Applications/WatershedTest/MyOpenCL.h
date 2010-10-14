/***********************************
	CTIMER
************************************/

#ifndef _MYOPENCL
#define _MYOPENCL

#ifdef MACINTOSH
#include <OpenCL/opencl.h>
#include <typeinfo>
#define max(a,b) (a<b?b:a)
#define min(a,b) (a<b?a:b)
#define __int64 long long
//#define PLATFORM_TYPE CL_DEVICE_TYPE_CPU
#define PLATFORM_TYPE CL_DEVICE_TYPE_ALL

#else
#include <CL/cl.h>
#include <stdio.h>
//#define PLATFORM_TYPE CL_DEVICE_TYPE_GPU
#define PLATFORM_TYPE CL_DEVICE_TYPE_ALL
#endif

class MyOpenCL {
public:
	char *szNLMProgram;
	cl_command_queue queue;
	cl_context context;
	//cl_device_id *devices;
	cl_device_id device;
	bool bInitialized;

	char* ReadNLMProgram(const char *szFilename) {
		if(szNLMProgram) {
			delete szNLMProgram;
			szNLMProgram = NULL;
		}
		FILE *fr;
		if(NULL == (fr = fopen(szFilename, "rb"))) {
			return NULL;
		}

		fseek(fr, 0, SEEK_END);
		int iFileLen = ftell(fr);

		fseek(fr, 0, SEEK_SET);

		szNLMProgram = new char[iFileLen + 1];
		fread(szNLMProgram, sizeof(char), iFileLen, fr);
		szNLMProgram[iFileLen] = 0;

		fclose(fr);
		return (iFileLen != 0)?szNLMProgram:NULL;
	}

	static bool PrintAvailableDevices(bool bAdvancedInfo) {
		cl_platform_id clSelectedPlatformID[10];
		cl_uint numPlatforms = 0;
		int errcode = clGetPlatformIDs(10, clSelectedPlatformID, &numPlatforms);
		if(errcode != CL_SUCCESS) {
			printf("Error getting OpenCL platforms.\n");
			return false;
		}

		printf("OpenCL devices:\n");

		char buff[4096];

		for(int platformId = 0; platformId < (int)numPlatforms; platformId++) {
			errcode = clGetPlatformInfo (clSelectedPlatformID[platformId], CL_PLATFORM_NAME, sizeof(buff), buff, NULL);
			if(errcode != CL_SUCCESS) continue;
			printf(" Platform %d: %s\n", platformId, buff);

			cl_uint ciDeviceCount;
			errcode = clGetDeviceIDs(clSelectedPlatformID[platformId], PLATFORM_TYPE, 0, NULL, &ciDeviceCount);
			if(errcode != CL_SUCCESS || ciDeviceCount < 1) {
				printf("Error getting devices for platform %d.\n", platformId);
				continue;
			}

			cl_device_id *devices = NULL;
			devices = new cl_device_id[ciDeviceCount];

			errcode = clGetDeviceIDs(clSelectedPlatformID[platformId], PLATFORM_TYPE, ciDeviceCount, devices, &ciDeviceCount);
			if(errcode != CL_SUCCESS || ciDeviceCount < 1) {
				printf("Error getting devices for platform %d.\n", platformId);
				continue;
			}

			for(int deviceId = 0; deviceId < (int)ciDeviceCount; deviceId++) {
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_NAME, sizeof(buff), buff, NULL);
				printf("  Device %d: \t%s\n", deviceId, buff);

				if(bAdvancedInfo == false)
					continue;

				clGetDeviceInfo(devices[deviceId], CL_DEVICE_VENDOR, sizeof(buff), buff, NULL);
				printf("   Device vendor: \t%s\n", buff);
				clGetDeviceInfo(devices[deviceId], CL_DRIVER_VERSION, sizeof(buff), buff, NULL);
				printf("   Driver version: \t%s\n", buff);

				cl_device_type type;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
				if(type & CL_DEVICE_TYPE_CPU)
					printf("   Device type: \tCPU\n");
				if(type & CL_DEVICE_TYPE_GPU)
					printf("   Device type: \tGPU\n");
				if(type & CL_DEVICE_TYPE_ACCELERATOR)
					printf("   Device type: \tACCELERATOR\n");
				if(type & CL_DEVICE_TYPE_DEFAULT)
					printf("   Device type: \tDEFAULT\n");

				cl_uint num;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(num), &num, NULL);
				printf("    CL_DEVICE_MAX_COMPUTE_UNITS: \t%d\n", num);

				clGetDeviceInfo(devices[deviceId], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(num), &num, NULL);
				printf("    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: %d\n", num);

				size_t workitem_size[3];
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &workitem_size, NULL);
				printf("    CL_DEVICE_MAX_WORK_ITEM_SIZES: \t%d %d %d\n", workitem_size[0], workitem_size[1], workitem_size[2]);
    
				size_t workgroup_size;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(workgroup_size), &workgroup_size, NULL);
				printf("    CL_DEVICE_MAX_WORK_GROUP_SIZE: \t%d\n", workgroup_size);

				cl_uint clock_frequency;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_frequency), &clock_frequency, NULL);
				printf("    CL_DEVICE_MAX_CLOCK_FREQUENCY: \t%d MHz\n", clock_frequency);

				cl_uint addr_bits;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_ADDRESS_BITS, sizeof(addr_bits), &addr_bits, NULL);
				printf("    CL_DEVICE_ADDRESS_BITS: \t\t%d\n", addr_bits);

				cl_bool image_support;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_IMAGE_SUPPORT, sizeof(image_support), &image_support, NULL);
				printf("    CL_DEVICE_IMAGE_SUPPORT: \t\t%s\n", image_support==CL_TRUE?"true":"false");

				cl_uint max_read_image_args;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(max_read_image_args), &max_read_image_args, NULL);
				printf("    CL_DEVICE_MAX_READ_IMAGE_ARGS: \t%d\n", max_read_image_args);

				cl_uint max_write_image_args;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(max_write_image_args), &max_write_image_args, NULL);
				printf("    CL_DEVICE_MAX_WRITE_IMAGE_ARGS: \t%d\n", max_write_image_args);
    
				size_t image2d_max_width;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(image2d_max_width), &image2d_max_width, NULL);
				size_t image2d_max_height;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(image2d_max_height), &image2d_max_height, NULL);
				size_t image3d_max_width;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(image3d_max_width), &image3d_max_width, NULL);
				size_t image3d_max_height;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(image3d_max_height), &image3d_max_height, NULL);
				size_t image3d_max_depth;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(image3d_max_depth), &image3d_max_depth, NULL);
				printf("    CL_DEVICE_IMAGE_MAX_WIDTH: \t\t2D (%d, %d), 3D (%d, %d, %d)\n", image2d_max_width, image2d_max_height, image3d_max_width, image3d_max_height, image3d_max_depth);

				cl_ulong max_mem_alloc_size;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc_size), &max_mem_alloc_size, NULL);
				printf("    CL_DEVICE_MAX_MEM_ALLOC_SIZE: \t%d MByte\n", (unsigned int)(max_mem_alloc_size / (1024 * 1024)));

				cl_ulong mem_size;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
				printf("    CL_DEVICE_GLOBAL_MEM_SIZE: \t\t%d MByte\n", (unsigned int)(mem_size / (1024 * 1024)));

				cl_bool error_correction_support;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(error_correction_support), &error_correction_support, NULL);
				printf("    CL_DEVICE_ERROR_CORRECTION_SUPPORT: %s\n", error_correction_support==CL_TRUE?"true":"false");

				cl_device_local_mem_type local_mem_type;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
				printf("    CL_DEVICE_LOCAL_MEM_TYPE: \t\t%s\n", local_mem_type == 1 ? "local" : "global");

				clGetDeviceInfo(devices[deviceId], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, NULL);
				printf("    CL_DEVICE_LOCAL_MEM_SIZE: \t\t%d KByte\n", (unsigned int)(mem_size / (1024)));

				clGetDeviceInfo(devices[deviceId], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(mem_size), &mem_size, NULL);
				printf("    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%d KByte\n", (unsigned int)(mem_size / (1024)));

				cl_command_queue_properties queue_properties;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_QUEUE_PROPERTIES, sizeof(queue_properties), &queue_properties, NULL);
				if( queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE )
					printf("    CL_DEVICE_QUEUE_PROPERTIES: \tCL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE\n");
				if( queue_properties & CL_QUEUE_PROFILING_ENABLE )
					printf("    CL_DEVICE_QUEUE_PROPERTIES: \tCL_QUEUE_PROFILING_ENABLE\n");

				clGetDeviceInfo(devices[deviceId], CL_DEVICE_EXTENSIONS, sizeof(buff), &buff, NULL);
				printf("    CL_DEVICE_EXTENSIONS: %s\n", buff);

				cl_uint vec_width_char;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(vec_width_char), &vec_width_char, NULL);
				cl_uint vec_width_short;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(vec_width_short), &vec_width_short, NULL);
				cl_uint vec_width_int;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(vec_width_int), &vec_width_int, NULL);
				cl_uint vec_width_long;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(vec_width_long), &vec_width_long, NULL);
				cl_uint vec_width_float;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(vec_width_float), &vec_width_float, NULL);
				cl_uint vec_width_double;
				clGetDeviceInfo(devices[deviceId], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(vec_width_double), &vec_width_double, NULL);
				printf("    CL_DEVICE_PREFERRED_VECTOR_WIDTH:\tchar %d, short %d, int %d, long %d, float %d, double %d\n", vec_width_char, vec_width_short, vec_width_int, vec_width_long, vec_width_float, vec_width_double);

			}

			if(devices != NULL)
				delete[] devices;
		}

		return true;
	}

	bool InitOpenCL(int platformId, int deviceId) {
/*		if(deviceId != 0) {
			printf("Device selection not implemented yet.\n");
			return false;
		}*/
		cl_platform_id clSelectedPlatformID[10]; 
		cl_uint numPlatforms = 0;
		int errcode = clGetPlatformIDs (10, clSelectedPlatformID, &numPlatforms);
		if(errcode != CL_SUCCESS)
			return false;

		if((int)numPlatforms <= platformId) {
			printf("Wrong platform ID.\n");
			return false;
		}

		cl_context_properties properties[3];
		properties[0] = CL_CONTEXT_PLATFORM;
		properties[1] = (cl_context_properties)clSelectedPlatformID[platformId];
		properties[2] = 0;
		context = clCreateContextFromType(properties, PLATFORM_TYPE, NULL, NULL, &errcode);
		if(context == NULL) {
			printf("Cannot create context\n");
			return false;
		}
		
		cl_uint ciDeviceCount;
		clGetDeviceIDs(clSelectedPlatformID[platformId], PLATFORM_TYPE, 0, NULL, &ciDeviceCount);

		if((int)ciDeviceCount <= deviceId) {
			printf("Wrong device ID.\n");
			clReleaseContext(context);
			return false;
		}
		
		cl_device_id *devices = new cl_device_id[ciDeviceCount];
		clGetDeviceIDs(clSelectedPlatformID[platformId], PLATFORM_TYPE, ciDeviceCount, devices, &ciDeviceCount);

		device = devices[deviceId];
		
		delete devices;

		queue = clCreateCommandQueue(context, device, 0, NULL);
		if(queue == NULL) {
			printf("Cannot create queue\n");
			clReleaseContext(context);
			return false;
		}
		bInitialized = true;
		
		size_t maxWGSize = 0;
		clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE , sizeof(size_t), &maxWGSize, NULL);
//		printf("CL_DEVICE_MAX_WORK_GROUP_SIZE = %d\n", maxWGSize);
		
		return true;
	}

	void DestroyOpenCL() {
		if(bInitialized) {
			clReleaseCommandQueue(queue);

			clReleaseContext(context);
		}
	}

public:
	MyOpenCL() {
		szNLMProgram = NULL;
		bInitialized = false;
	}

	~MyOpenCL() {
		DestroyOpenCL();
		if(szNLMProgram != NULL)
			delete szNLMProgram;
		bInitialized = false;
	}
};

#endif

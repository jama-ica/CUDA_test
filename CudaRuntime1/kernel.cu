
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>	  // srand,rand
#include <cmath>
#include <bitset>

static const unsigned char REEL_COUNT = 5;
static const unsigned char REEL_LEN = 128;
static const unsigned char WINDOW_SIZE = 3;

static const unsigned char SYMBOL_KIND = 15;

static const int RESULT_SIZE = 10000;

cudaError_t checkFullSymbolWithCuda(const char* reel1, const char* reel2, const char* reel3, const char* reel4, const char* reel5, unsigned char reel_len, unsigned char window_size, char* result, unsigned int result_size);


__global__ void checkFullSymbolKernel(const char* reel1, const char* reel2, const char* reel3, const char* reel4, const char* reel5, unsigned char reel_len, unsigned char window_size, char* result, unsigned int result_size)
{
	__int64 index = (__int64)blockIdx.x * (__int64)blockDim.x + (__int64)threadIdx.x;

	int reel_pos[REEL_COUNT] = {};

	// reel 5 pos
	__int64 origin = index;
	__int64 shifted = origin >> 7;
	reel_pos[4] = origin - (shifted << 7);

	origin = shifted;
	shifted = origin >> 7;
	reel_pos[3] = origin - (shifted << 7);
	
	origin = shifted;
	shifted = origin >> 7;
	reel_pos[2] = origin - (shifted << 7);

	origin = shifted;
	shifted = origin >> 7;
	reel_pos[1] = origin - (shifted << 7);

	reel_pos[0] = shifted;

	char symbol_flg[SYMBOL_KIND] = {};
	int symbol = 0;
	for(int i = 0 ; i < REEL_COUNT; i++)
	{ 
		for (int j = 0 ; j < WINDOW_SIZE; j++)
		{
			int pos = reel_pos[i] - 1 + j;
			if (0 > pos) {
				pos += REEL_LEN;
			}
			else if (pos >= REEL_LEN) {
				pos -= REEL_LEN;
			}

			switch (i)
			{
			case 0:	symbol = reel1[pos]; break;
			case 1:	symbol = reel2[pos]; break;
			case 2:	symbol = reel3[pos]; break;
			case 3:	symbol = reel4[pos]; break;
			case 4:	symbol = reel5[pos]; break;
			}
			symbol_flg[symbol] = 1;
		}
	}

	bool full_symbol = true;
	for (int i = 0; i < SYMBOL_KIND; i++)
	{
		if (symbol_flg[i] == 0)
		{
			full_symbol = false;
			break;
		}
	}
	if(full_symbol)
	{
		int res_i = index % RESULT_SIZE;
		for (int i = 0; i < REEL_COUNT; i++)
		{
			result[res_i* REEL_COUNT+i] = reel_pos[i];
		}
	}
}

int main()
{
	char Reel1[REEL_LEN] = {};
	char Reel2[REEL_LEN] = {};
	char Reel3[REEL_LEN] = {};
	char Reel4[REEL_LEN] = {};
	char Reel5[REEL_LEN] = {};
	for (int i = 0; i < REEL_LEN; i++)
	{
		Reel1[i] = rand() % SYMBOL_KIND;
		Reel2[i] = rand() % SYMBOL_KIND;
		Reel3[i] = rand() % SYMBOL_KIND;
		Reel4[i] = rand() % SYMBOL_KIND;
		Reel5[i] = rand() % SYMBOL_KIND;
	}

	char Result[RESULT_SIZE * REEL_COUNT] = {};

	printf("Start\n");

	// Add vectors in parallel.
	cudaError_t cudaStatus = checkFullSymbolWithCuda(Reel1, Reel2, Reel3, Reel4, Reel5, REEL_LEN, WINDOW_SIZE, Result, RESULT_SIZE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "checkFullSymbolWithCuda failed!");
		return 1;
	}

	for ( int i = 0 ; i < RESULT_SIZE * REEL_COUNT ; i += REEL_COUNT )
	{
		bool isEmpty = true;
		for (int j = 0 ; j < REEL_COUNT ; j++)
		{
			if (0 != Result[i+j])
			{
				isEmpty = false;
				break;
			}
		}
		if (!isEmpty)
		{
			printf("%d: {", i/ REEL_COUNT);
			for (int j = 0 ; j < REEL_COUNT ; j++)
			{
				printf("%d,", Result[i+j]);
			}
			printf("} \n");

			for (int w = 0; w < WINDOW_SIZE; w++)
			{
				printf("[");
				for (int r = 0; r < REEL_COUNT; r++)
				{ 
					int pos = Result[i + r] - 1 + w;
					if (0 > pos) {
						pos += REEL_LEN;
					}
					else if (pos >= REEL_LEN) {
						pos -= REEL_LEN;
					}
					int symbol = 0;
					switch (r)
					{
					case 0:	symbol = Reel1[pos]; break;
					case 1:	symbol = Reel2[pos]; break;
					case 2:	symbol = Reel3[pos]; break;
					case 3:	symbol = Reel4[pos]; break;
					case 4:	symbol = Reel5[pos]; break;
					}
					printf("%d,", symbol);

				}
				printf("]\n");
			}
		}
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	printf("Finished\n");
	return 0;
}

cudaError_t checkFullSymbolWithCuda(const char* reel1, const char* reel2, const char* reel3, const char* reel4, const char* reel5, unsigned char reel_len, unsigned char window_size, char* result, unsigned int result_size)
{
	printf("checkFullSymbolWithCuda: Start\n");

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	char* dev_reel1 = 0;
	char* dev_reel2 = 0;
	char* dev_reel3 = 0;
	char* dev_reel4 = 0;
	char* dev_reel5 = 0;

	// Allocate GPU buffers for three vectors (two input, one output)	.
	cudaStatus = cudaMalloc((void**)&dev_reel1, reel_len * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_reel2, reel_len * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_reel3, reel_len * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_reel4, reel_len * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_reel5, reel_len * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	char* dev_result = 0;

	cudaStatus = cudaMalloc((void**)&dev_result, result_size * REEL_COUNT * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_reel1, reel1, reel_len * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_reel2, reel2, reel_len * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_reel3, reel3, reel_len * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_reel4, reel4, reel_len * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_reel5, reel5, reel_len * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_result, result, result_size * REEL_COUNT * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// N = 128^5 = 34359738368
	// 34359738368/1024 = 33554432
	// 
	// MAX blockPerGrid = 2147483647
	// Max Threads = 1024*2147483647 = 2199023254528
	__int64 N = std::pow(REEL_LEN, REEL_COUNT);

	__int64 threadsPerBlock = 1024;
	__int64 blocksPerGrid = N / threadsPerBlock; //33554432;

	// Launch a kernel on the GPU with one thread for each element.
	printf("checkFullSymbolKernel: Start N=%d, threadsPerBlock=%d, blocksPerGrid=%d\n", N, threadsPerBlock, blocksPerGrid);
	checkFullSymbolKernel <<<blocksPerGrid, threadsPerBlock >>> (dev_reel1, dev_reel2, dev_reel3, dev_reel4, dev_reel5, REEL_LEN, WINDOW_SIZE, dev_result, RESULT_SIZE);
	printf("checkFullSymbolKernel: End\n");

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "checkFullSymbolKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching checkFullSymbolKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(result, dev_result, result_size * REEL_COUNT * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_reel1);
	cudaFree(dev_reel2);
	cudaFree(dev_reel3);
	cudaFree(dev_reel4);
	cudaFree(dev_reel5);
	cudaFree(dev_result);

	printf("checkFullSymbolWithCuda: End\n");
	return cudaStatus;
}


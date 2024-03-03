
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

cudaError_t containsAllSymbolsWithCuda(const char reel[REEL_COUNT][REEL_LEN], char* result);

__global__ void containsAllSymbolsKernel(const char* reel1, const char* reel2, const char* reel3, const char* reel4, const char* reel5, char* result)
{
	__int64 index = (__int64)blockIdx.x * (__int64)blockDim.x + (__int64)threadIdx.x;

	char reel_pos[REEL_COUNT] = {};

	// calc reel pos
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

	// check symbols
	bool symbol_flg[SYMBOL_KIND] = {false};
	char symbol = 0;
	for(char r = 0 ; r < REEL_COUNT; r++)
	{
		for (char w = 0 ; w < WINDOW_SIZE; w++)
		{
			char pos = reel_pos[r] - 1 + w;
			if (0 > pos) {
				pos += REEL_LEN;
			}
			else if (pos >= REEL_LEN) {
				pos -= REEL_LEN;
			}

			switch (r)
			{
			case 0:	symbol = reel1[pos]; break;
			case 1:	symbol = reel2[pos]; break;
			case 2:	symbol = reel3[pos]; break;
			case 3:	symbol = reel4[pos]; break;
			case 4:	symbol = reel5[pos]; break;
			}

			if (r == 1 && symbol_flg[symbol])
			{
				return; // 当選は除外
			}

			symbol_flg[symbol] = true;
		}
	}

	bool containsAllSymbols = true;
	for (int i = 0; i < SYMBOL_KIND; i++)
	{
		if (!symbol_flg[i])
		{
			containsAllSymbols = false;
			break;
		}
	}
	if(containsAllSymbols)
	{
		// set reel pos to result
		int res_i = index % RESULT_SIZE;
		for (char i = 0; i < REEL_COUNT; i++)
		{
			result[res_i* REEL_COUNT + i] = reel_pos[i];
		}
	}
}

int main()
{
	printf("Start\n");

	char Reels[REEL_COUNT][REEL_LEN] = {{}};
	for (int r = 0; r < REEL_COUNT; r++)
	{ 
		for (int w = 0; w < REEL_LEN; w++)
		{
			Reels[r][w] = rand() % SYMBOL_KIND;
		}
	}

	char Result[RESULT_SIZE * REEL_COUNT] = {-1};

	// Add vectors in parallel.
	cudaError_t cudaStatus = containsAllSymbolsWithCuda(Reels, Result);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "containsAllSymbolsWithCuda failed!");
		return 1;
	}

	for ( int i = 0 ; i < RESULT_SIZE ; i++ )
	{
		int index = i * REEL_COUNT;
		bool isEmpty = (Result[index] == -1);
		if (isEmpty)
		{
			continue;
		}
		printf("%d: {", i);
		for (int j = 0 ; j < REEL_COUNT ; j++)
		{
			printf("%d,", Result[index + j]);
		}
		printf("} \n");

		for (int w = 0; w < WINDOW_SIZE; w++)
		{
			printf("[");
			for (int r = 0; r < REEL_COUNT; r++)
			{ 
				int pos = Result[index + r] - 1 + w;
				if (0 > pos) {
					pos += REEL_LEN;
				}
				else if (pos >= REEL_LEN) {
					pos -= REEL_LEN;
				}
				int symbol = Reels[r][pos];
				printf("%d,", symbol);
			}
			printf("]\n");
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

cudaError_t containsAllSymbolsWithCuda(const char reel[REEL_COUNT][REEL_LEN], char* result)
{
	printf("%s: Start\n", __FUNCTION__);

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	char* dev_reels[REEL_COUNT] = {};

	// Allocate GPU buffers for three vectors (two input, one output)
	for(int i = 0 ; i < REEL_COUNT ; i++)
	{
		cudaStatus = cudaMalloc((void**)&(dev_reels[i]), REEL_LEN * sizeof(char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
	}

	char* dev_result = 0;

	cudaStatus = cudaMalloc((void**)&dev_result, RESULT_SIZE * REEL_COUNT * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	for (int i = 0; i < REEL_COUNT; i++)
	{
		cudaStatus = cudaMemcpy(dev_reels[i], reel[i], REEL_LEN * sizeof(char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
	}

	cudaStatus = cudaMemcpy(dev_result, result, RESULT_SIZE * REEL_COUNT * sizeof(char), cudaMemcpyHostToDevice);
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
	printf("containsAllSymbolsKernel: Start N=%d, threadsPerBlock=%d, blocksPerGrid=%d\n", N, threadsPerBlock, blocksPerGrid);
	containsAllSymbolsKernel <<<blocksPerGrid, threadsPerBlock >>> (dev_reels[0], dev_reels[1], dev_reels[2], dev_reels[3], dev_reels[4], dev_result);
	printf("containsAllSymbolsKernel: End\n");

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
	cudaStatus = cudaMemcpy(result, dev_result, RESULT_SIZE * REEL_COUNT * sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	for (int i = 0; i < REEL_COUNT; i++)
	{
		cudaFree(dev_reels[i]);
	}
	cudaFree(dev_result);

	printf("%s: Ebd\n", __FUNCTION__);
	return cudaStatus;
}


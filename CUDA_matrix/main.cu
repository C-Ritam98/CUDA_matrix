#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <chrono>
#include <fstream>
using namespace std;

#define TIME_NOW std::chrono::high_resolution_clock::now()
#define TIME_DIFF(gran, start, end) std::chrono::duration_cast<gran>(end - start).count()


#include "gpu_thread.h"


void reference(int N, int *matA, int *matB, int *output)
{
    
    for(int i = 0; i < N; ++i) {
        int temp = 0;
       
        for(int j = 0; j < i + 1; ++j) {
            int rowA = j;
            int colA = i - j;
            int rowB = i - j;
            int colB = N - j - 1;
            temp += matA[rowA * N + colA] * matB[rowB * N + colB];
        }
        output[i] = temp;
    }
    
    
    for(int i = N; i < 2 * N - 1; ++i) {
        int temp = 0;
        
        for(int j = 0; j < 2 * N - (i + 1); ++j) {
            int rowA = i + 1 + j - N;
            int colA = N - j - 1;
            int rowB = N - j - 1;
            int colB = 2 * N - j - 2 - i;
            temp += matA[rowA * N + colA] * matB[rowB * N + colB];
        }
        output[i] = temp;
    }
}

int main(int argc, char *argv[])
{
    
    int N;
    string file_name; 
    if (argc < 2) 
        file_name = "data/input_16384.in"; 
    else 
        file_name = argv[1]; 
    ifstream input_file; 
    input_file.open(file_name); 
    input_file >> N;
    cout << "Input matrix of size " << N << "\n";
    
    
    int *matA = new int[N * N];
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            input_file >> matA[i * N + j];

    
    int *matB = new int[N * N];
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            input_file >> matB[i * N + j];
    
    
    int *output_reference = new int[2 * N - 1];
    auto begin = TIME_NOW;
    reference(N, matA, matB, output_reference);
    auto end = TIME_NOW;
    cout << "Reference execution time: " << (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n"; 
    
    
    int *output_single = new int[2 * N - 1];
    begin = TIME_NOW;
    gpuThread(N, matA, matB, output_single);
    end = TIME_NOW;
    cout << "GPU execution time: " << (double)TIME_DIFF(std::chrono::microseconds, begin, end) / 1000.0 << " ms\n"; 
    
    for(int i = 0; i < 2 * N - 1; ++i)
        if(output_single[i] != output_reference[i]) {
            cout << "Mismatch at " << i << "\n";
            cout << "GPU output: " << output_single[i] << ", required output: " << output_reference[i] << "\n";
            exit(0);
        }
    input_file.close(); 
}

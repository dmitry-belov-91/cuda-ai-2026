# Matrix Multiplication using cuBLAS

### Prerequisites

```
sudo apt install nvidia-cuda-toolkit
```

### Build

```
nvcc main.cu gemm_cublas.cu -o gemm_cublas -lcublas
```


### Run
```
./gemm_cublas [number of elements]
```

### Results
Results for 1000 elements:
* Intel(R) Xeon(R) w5-3425, 12 cores, 3.20 GHz
* NVIDIA GeForce RTX 5090

```
 ./gemm_cublas 1000
Elements num (^2): 1000000
------------------------
Naive GEMM: 3866.53 ms
GEMM CUBLAS (no prealloccation): 2.38257 ms
        Mean abs diff: 0.00041021176222
        Max diff: 0.006103515625
GEMM CUBLAS: 1.62617695332 ms
        Mean abs diff: 0.00041021176222
        Max diff: 0.006103515625
```

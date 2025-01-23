# GPT-2 병렬 최적화를 위한 고성능 컴퓨팅 프로젝트

본 프로젝트는 high performance를 위해 다중 GPU 및 노드를 활용해 C language로 작성된 GPT-2 텍스트 생성 모델을 병렬화하고 최적화한 작업을 포함하고 있습니다.
서울대학교 2024-1 **"확장형 고성능 컴퓨팅 및 데이터 사이언스를 위한 컴퓨팅"** (이재진 교수님) 수업의 final project입니다.

## Project Overview

본 프로젝트의 목표는 GPT-2 125M 모델의 sequential code를 최적화:
- **Parallelization**: 4개의 노드(총 16 GPU)에서 병렬화 구현.
- **Throughput Optimization**: 초당 생성 토큰 수(Throughput)를 향상.
- **Memory Management**: 효율적인 메모리 사용과 데이터 이동 최소화.
- **Nsight Profiling**: 병목 지점을 확인하기 위한 프로파일링.
- CUDA, OpenMP, MPI를 활용한 고성능 병렬 컴퓨팅 기법 적용.

---

## Performance Summary
```
===================================================
Model: GPT-2 125M
---------------------------------------------------
Validation: ON
Number of Prompts: 6400
Number of Tokens to generate: 8
Input binary path: ./data/input.bin
Model parameter path: /shpc24/project_model_paramet
Answer binary path: ./data/answer.bin
Output binary path: ./data/output.bin
===================================================

Initializing input and parameters...Done
Generating tokens...Done!
Elapsed time: 116.649317 (sec)
Throughput: 438.922417 (tokens/sec)
Finalizing...Done
Saving output to ./data/output.bin...Done
Validation...PASS
salloc: Relinquishing job allocation 746174

```
<br>

- **Batch Size**: 800
- **Number of Prompts**: 6400
- **Throughput**: 최대 **438.9 tokens/second** 달성.

---

## Optimization

### Tensor Structure
![image](https://github.com/user-attachments/assets/80e0f913-f840-4e52-b399-f36d5431ad4d)

- **Initial Issue**: Nsight 프로파일링 결과, 불필요한 host-to-device 메모리 전송이 자주 발생하는 것을 확인.
- **Solution**: Tensor 구조체의 `buf`를 device 상에 유지하도록 설계하여, 불필요한 메모리 전송 제거 및 성능 향상.<br><br>
  ```C
  Tensor::Tensor(const vector<size_t> &shape_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) { 
        shape[i] = shape[i]; 
    }
    size_t N_ = num_elem();

    cudaError_t err = cudaMallocManaged(&buf, N_ * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate unified memory for Tensor: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    host_buf = (float*)malloc(N_ * sizeof(float));
    if (host_buf == nullptr) {
        fprintf(stderr, "Failed to allocate host memory for Tensor\n");
        cudaFree(buf);
        exit(EXIT_FAILURE);
    }

    memset(buf, 0, N_ * sizeof(float));
    memset(host_buf, 0, N_ * sizeof(float));
  }

  ```

### MPI Communication
- **기능**: `generate_tokens` 함수에 **MPI**를 적용해 다수의 프로세서 코어에 작업을 분산시킴.
- **구현**:
  - `MPI_Iscatterv`를 사용해 입력 데이터를 각 프로세스에 분산.<br>
    ```C
    void generate_tokens(int *input, int *output, size_t n_prompt, size_t n_token) {

    int mpi_rank, mpi_size;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    std::vector<int> sendcounts(mpi_size);
    std::vector<int> displs(mpi_size);

    int sum = 0;
    for (int i = 0; i < mpi_size; ++i) {
        sendcounts[i] = (n_prompt / mpi_size + (i < n_prompt % mpi_size ? 1 : 0)) * tokens_per_prompt;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    std::vector<int> local_input(sendcounts[mpi_rank]);

    MPI_Request scatter_request;
    MPI_Iscatterv(input, sendcounts.data(), displs.data(), MPI_INT, local_input.data(), sendcounts[mpi_rank], MPI_INT, ...);

    size_t local_n_prompt = sendcounts[mpi_rank] / tokens_per_prompt;
    size_t batch_size = BATCH_SIZE;
    std::vector<int> local_output(local_n_prompt * n_token);

    MPI_Wait(&scatter_request, MPI_STATUS_IGNORE);
    
    ...
    }

    ```
  - **OpenMP**를 활용하여 각 프로세스 내부에서 병렬적으로 토큰 생성.<br><br>
    ```C
    std::vector<int> next_tokens(actual_batch_size);

    top1_sampling(logit_a, next_tokens.data(), tokens_per_prompt, actual_batch_size);
    
    #pragma omp parallel for
    for (size_t b = 0; b < actual_batch_size; b++) {
        input_prompts[b].push_back(next_tokens[b]);
        size_t local_output_index = (p + b) * n_token + t;
        local_output[local_output_index] = next_tokens[b];
    }
    
    prompt_size += 1;
    
    free_activations();
    ```
  - 최종 출력을 `MPI_Igatherv`로 다시 모음.<br><br>
    ```C
    int local_output_size = local_n_prompt * n_token;
    std::vector<int> recvcounts(mpi_size);
    std::vector<int> displs_output(mpi_size);
    
    if (mpi_rank == 0) {
        int output_sum = 0;
        for (int i = 0; i < mpi_size; ++i) {
            recvcounts[i] = (n_prompt / mpi_size + (i < n_prompt % mpi_size ? 1 : 0)) * n_token;
            displs_output[i] = output_sum;
            output_sum += recvcounts[i];
        }
    }
    
    MPI_Request gather_request;
    MPI_Igatherv(local_output.data(), local_output_size, MPI_INT, output, recvcounts.data(), displs_output.data(), MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Wait(&gather_request, MPI_STATUS_IGNORE);
    ```
- **효과**: 대규모 데이터 처리에서 효율성과 확장성 극대화.
  

### Matmul Optimization
- **CUDA Streams 및 Shared Memory**:
  - **Shared Memory**를 사용해 글로벌 메모리 접근을 최소화하여 성능 향상.
  - **CUDA Streams**를 활용해 비동기적으로 배치를 처리, GPU 자원 활용도 극대화.
- **커스텀 구현**:
  - `matmul`: 일반적인 행렬곱 연산.
  - `matmul_2`: Transpose 및 Scaling 연산을 통합하여 메모리 접근 패턴 최적화.
  - `matmul_3`: 메모리 접근을 더욱 개선하고, 비동기 연산으로 처리 속도 향상.<br><br>
  ```C
  __global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        if (row < M && t * BLOCK_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * BLOCK_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
  }
  ```
  ```C
  void matmul(Tensor *in1, Tensor *in2, Tensor *out, size_t batch_size, bool verbose) {
    size_t M = in1->shape[1];
    size_t K = in1->shape[2];
    size_t N = in2->shape[2];

    float *d_in1 = in1->buf;
    float *d_in2 = in2->buf;
    float *d_out = out->buf;

    size_t size_out = batch_size * M * N * sizeof(float);

    CHECK_CUDA(cudaMemset(d_out, 0, size_out));

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    for (size_t b = 0; b < batch_size; b++) {
        int stream_idx = b % NUM_STREAMS;
        matmul_kernel<<<gridDim, blockDim, 0, streams[stream_idx]>>>(d_in1 + b * M * K, d_in2 + b * K * N, d_out + b * M * N, M, N, K);
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
  }
  ```

### Softmax Function
- **최적화 기법**:
  - **Shared Memory** 및 **Warp-level 함수**(`__shfl_down_sync`)를 활용해 메모리 접근 시간 단축.
  - **Synchronization Points** 최소화(`__syncthreads`)를 통해 불필요한 스레드 대기 시간 제거.
- **효과**: Softmax 연산의 효율성 대폭 향상.<br><br>
  ```C
  __global__ void softmax_kernel(float *inout, int s, int H) {
    extern __shared__ float shared_mem[];
    const int batch = blockIdx.z;
    const int row = blockIdx.y;
    const int col = threadIdx.x;
    const int idx = batch * s * H + row * H + col;

    // Load data to shared memory
    float x = (col < H) ? inout[idx] : -INFINITY;
    shared_mem[col] = x;
    __syncthreads();

    // Find the maximum value in the row
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float temp = __shfl_down_sync(0xffffffff, shared_mem[col], offset);
        shared_mem[col] = max(shared_mem[col], temp);
    }
    float max_val = shared_mem[0];

    // Compute the exponential values and sum
    x = (col < H) ? expf(inout[idx] - max_val) : 0.0f;
    shared_mem[col] = x;
    __syncthreads();
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        shared_mem[col] += __shfl_down_sync(0xffffffff, shared_mem[col], offset);
    }
    float sum = shared_mem[0];

    // Normalize
    if (col < H) {
        inout[idx] = x / sum;
    }
  }
  ```
  ```C
  void softmax(Tensor *inout, size_t batch_size) {
    size_t s = inout->shape[1];
    size_t H = inout->shape[2];

    dim3 blockDim(H);
    dim3 gridDim(1, s, batch_size);
    size_t shared_mem_size = H * sizeof(float);

    softmax_kernel<<<gridDim, blockDim, shared_mem_size>>>(inout->buf, s, H);
    CHECK_CUDA(cudaGetLastError());
  }
  ```

---

## 구현 세부사항

### Workflow
1. **Initialization**:
   - MPI 초기화 및 rank와 process 정보 획득.
   - 입력 프롬프트를 각 프로세스에 분할.
2. **Token Generation**:
   - OpenMP로 배치 단위 프롬프트를 병렬 처리.
   - CUDA 최적화 커널로 행렬 연산 실행.
3. **Result Gathering**:
   - 각 프로세스의 출력 데이터를 root 프로세스로 수집.

### CUDA Kernel Highlights
- **Block and Thread Configuration**:
  - 최적화된 block size: `8 x 8` (실험을 통해 결정).
  - Grid 크기를 행렬의 크기에 맞춰 동적으로 설정.
- **Minimized Memory Transfers**:
  - Persistent memory buffer를 사용해 host-device 데이터 전송 최소화.
  - CUDA Streams를 통해 데이터 전송과 연산을 중첩시켜 처리.

---

## 학습 내용

- **Profiling**의 중요성: Nsight와 같은 도구를 활용해 병목현상을 파악하고 해결.
- **Memory Management**: 불필요한 host-to-device 전송을 줄여 성능을 크게 향상.
- **Parallel Programming**: MPI와 OpenMP를 활용한 확장성과 성능 향상.

---

## Project Structure
```
./
├── include/                 # Header files defining core structures and interfaces
│   ├── tensor.h             # Tensor structure definition
│   ├── layer.h              # Deep learning layer definitions
│   └── model.h              # GPT-2 model configuration and interfaces
├── src/                     # Source files implementing core functionalities
│   ├── tensor.cu            # Implementation of tensor operations
│   ├── layer.cu             # Implementation of deep learning layers
│   ├── model.cu             # GPT-2 model implementation and optimization
│   └── main.cpp             # Driver code for execution (do not modify)
├── data/                    # Directory for input/output data
├── Makefile                 # Build configuration file for compilation
├── run.sh                   # Script to run the project
├── README.md                # Readme file
└── report.pdf               # Detailed report on optimization strategies and results
```

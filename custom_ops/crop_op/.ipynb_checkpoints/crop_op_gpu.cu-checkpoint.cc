#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "crop_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void CropCudaKernel(
  const T* image_ptr,
  const int* crop_centers_ptr,
  const int image_size,
  const int channels,
  int crop_size,
  const int num_crops,
  T* crops_ptr
) {
    const int crop_id = blockIdx.x/crop_size;
    const int center_x = crop_centers_ptr[crop_id*3];
    const int center_y = crop_centers_ptr[1 + crop_id*3];
    const int center_z = crop_centers_ptr[2 + crop_id*3];
    int offset = (blockIdx.x % crop_size) * crop_size*crop_size*channels;
    for (int id = threadIdx.x; id < crop_size*crop_size*channels; id += blockDim.x) {
        // Coordinates inside the crop (0 <= coords < crop_size)
        int id_temp = offset + id;
        const int c = id_temp % channels;
        id_temp /= channels;
        const int z = id_temp % crop_size;
        id_temp /= crop_size;
        const int y = id_temp % crop_size;
        const int x = id_temp / crop_size;
         // Corresponding coordinates in original image
        int image_x = x + (center_x - crop_size / 2);
        int image_y = y + (center_y - crop_size / 2);
        int image_z = z + (center_z - crop_size / 2);
        int img_idx = c + channels * (image_z + image_size * (image_y + image_size * image_x ));

        if ((img_idx >= image_size * image_size * image_size * channels) || (img_idx < 0)) continue;

        int crop_idx = c + channels * (z + crop_size * (y + crop_size * (x + crop_size * crop_id)));
        crops_ptr[crop_idx] = image_ptr[img_idx];
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void CropFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d,
    const T* image_ptr,
    const int* crop_centers_ptr,
    int crop_size,
    int image_size,
    int channels,
    int num_crops,
    T* crops_ptr
  ) {
  // Launch the cuda kernel.
  int block_count = num_crops;
  int thread_per_block = 1024;
  CropCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(
        image_ptr,
        crop_centers_ptr,
        image_size,
        channels,
        crop_size,
        num_crops,
        crops_ptr
      );
    cudaDeviceSynchronize();
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct CropFunctor<GPUDevice, float>;
template struct CropFunctor<GPUDevice, int32>;

#endif // GOOGLE_CUDA
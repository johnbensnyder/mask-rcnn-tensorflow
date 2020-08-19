import tensorflow as tf
import numpy as np
import time

def crop(patch_centers, N, data):
    """
    Slice patches of size N centered at patch_centers in data.
    Assumes data has shape (1, M, M, M, channels)
    or (1, M, M, channels)
    """
    coords0 = np.floor(patch_centers - N/2.0)  # bottom left corner
    coords1 = np.floor(patch_centers + N/2.0)  # top right corner
    dim = patch_centers.shape[1]
    image_size = data.shape[1]
    coords0 = np.clip(coords0, 0, image_size).astype(int)
    coords1 = np.clip(coords1, 0, image_size).astype(int)
    crops = np.zeros((coords0.shape[0],) + (N,) * dim + (data.shape[-1],))
    crops_labels = np.zeros_like(crops)
    for j in range(len(coords0)):
        padding = []
        for d in range(dim):
            pad = np.maximum(N - (coords1[j, d] - coords0[j, d]), 0)
            if coords0[j, d] == 0.0:
                padding.append((pad, 0))
            else:
                padding.append((0, pad))
        padding.append((0, 0))
        if dim == 2:
            crops[j] = np.pad(data[0,
                                   coords0[j, 0]:coords1[j, 0],
                                   coords0[j, 1]:coords1[j, 1],
                                   :],
                              padding, 'constant')
        else:  # dim == 3
            crops[j] = np.pad(data[0,
                                   coords0[j, 0]:coords1[j, 0],
                                   coords0[j, 1]:coords1[j, 1],
                                   coords0[j, 2]:coords1[j, 2],
                                   :],
                              padding, 'constant')
    return crops

class CropTest(tf.test.TestCase):
    def testCrop(self):
        crop_module = tf.load_op_library('crop_op.so')

        np.random.seed(123)
        tf.set_random_seed(123)
        
        N = 192
        # The more steps here, the more accurate the timings will be
        # Remember that TF first few calls to sess.run are always slower
        MAX_STEPS = 200
        CROP_SIZE = 64
        
        # Define dummy crop centers
        image_np = (np.random.rand(N, N, N, 1) * N).astype(np.float32)
        crop_centers_np = np.random.randint(50, high=100, size=(100, 3))
        
        # Define TF equivalents
        image = tf.constant(image_np, dtype=tf.float32)
        crop_centers = tf.constant(crop_centers_np, dtype=tf.int32)
        
        # >>> Call our CUDA kernel! <<<
        crops = crop_module.crop(image, crop_centers, CROP_SIZE)
        
        with self.test_session():
            duration = 0
            for i in range(MAX_STEPS):
                start = time.time()
                tf_result = crops.eval()
                end = time.time()
                duration += end - start
            print("TF duration = %f s" % (duration / MAX_STEPS))
            duration = 0
            for i in range(MAX_STEPS):
                start = time.time()
                np_result, _ = crop_numpy(crop_centers_np, CROP_SIZE, image_np[np.newaxis, ...])
                end = time.time()
                duration += end - start
            print("NP duration = %f s" % (duration / MAX_STEPS))
            self.assertAllClose(tf_result, np_result)
            

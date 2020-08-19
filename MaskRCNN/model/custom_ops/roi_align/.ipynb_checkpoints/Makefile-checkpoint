TF_INC=/usr/local/lib/python3.6/dist-packages/tensorflow/include
TF_LIB=/usr/local/lib/python3.6/dist-packages/tensorflow

CC        = gcc -O2 -pthread
CXX       = g++
GPUCC     = nvcc
CFLAGS    = -std=c++11 -I$(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0
GPUCFLAGS = -c
LFLAGS    =  -shared -fPIC -ltensorflow_framework -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -L/usr/local/cuda/lib64 -I/usr/local/cuda/include
GPULFLAGS = -x cu -shared -Xcompiler -fPIC -ltensorflow_framework -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB)
DEBUG = -g -G
GPUDEF    = -D GOOGLE_CUDA=1
CGPUFLAGS = -lcudart


SRC       = roi_align_op.cc
GPUSRC    = roi_align_op.cu.cc
PROD      = roi_align_op.so
GPUPROD = roi_align_op_cu.so

default: gpu

cpu:
	$(CXX) $(CFLAGS) $(SRC) $(LFLAGS) -o $(PROD)

gpu:
	$(GPUCC) $(CFLAGS) $(GPUCFLAGS) $(GPUSRC) $(GPULFLAGS) -o $(GPUPROD) $(GPUDEF) -I/usr/local/ --expt-relaxed-constexpr -D_MWAITXINTRIN_H_INCLUDED
	$(CXX) $(CFLAGS)  $(SRC) $(GPUPROD) $(LFLAGS) $(CGPUFLAGS) -o $(PROD) $(GPUDEF)

clean:
	rm -f $(PROD) $(GPUPROD)

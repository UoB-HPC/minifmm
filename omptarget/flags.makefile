CC=clang++

ifeq ($(TARGET), GPU)
CFLAGS=-Ofast -mllvm --nvptx-f32ftz -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=$(ARCH) 
LIBS=-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=$(ARCH)
else 
CFLAGS=-Ofast -fopenmp -fopenmp-targets=x86_64 -march=$(ARCH) 
LIBS=-fopenmp -fopenmp-targets=x86_64 -march=$(ARCH) 
endif

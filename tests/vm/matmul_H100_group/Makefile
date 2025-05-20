# Compiler
NVCC?=nvcc
GPU=H100

TARGET=group_matmul
SRC=group_matmul.cu

# Python version configuration
# ifndef PYTHON_VERSION
# PYTHON_VERSION=3.11
# endif

THUNDERKITTENS_ROOT=$(shell pwd)/../../..
EXTENSION_SUFFIX=.cpython-311-x86_64-linux-gnu.so

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas -lineinfo
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype $(shell python -m pybind11 --includes) -shared -fPIC
# NVCCFLAGS+= -g -G

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
else
NVCCFLAGS+= -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(EXTENSION_SUFFIX)

# Clean target
clean:
	rm -f $(TARGET)$(EXTENSION_SUFFIX)

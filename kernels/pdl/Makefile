# GPU Selection (H100)
GPU_TARGET=H100

# Compiler
NVCC?=nvcc

# Compiler flags optimized for H100
NVCCFLAGS=-DNDEBUG \
    -Xcompiler=-fPIE \
    -Xcompiler -fopenmp \
    --expt-extended-lambda \
    --expt-relaxed-constexpr \
    -Xcompiler=-Wno-psabi \
    -Xcompiler=-fno-strict-aliasing \
    --use_fast_math \
    -forward-unknown-to-host-compiler \
    -O3 \
    -Xnvlink=--verbose \
    -Xptxas=--verbose \
    -Xptxas=--warn-on-spills \
    -std=c++20 \
    -MD -MT -MF \
    -x cu \
    -lrt -lpthread -ldl \
    -DKITTENS_HOPPER \
    -arch=sm_90a \
    -lcuda \
    -lcudadevrt \
    -lcudart_static \
    -lcublas \
    -lgomp

# Target and source files
TARGET=pdl
SRC=pdl.cu

# Default target
all: $(TARGET)

# Compilation rule
$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
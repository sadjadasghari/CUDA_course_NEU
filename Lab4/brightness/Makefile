# environment
SM := 35

# compilers
CUDA_PATH ?= $(shell test -d /shared/apps/cuda7.0 && echo /shared/apps/cuda7.0)
ifeq ($(CUDA_PATH),)
NVCC := nvcc
else
NVCC := $(CUDA_PATH)/bin/nvcc
endif

GCC := g++

# libraries
CUDA_LIB_PATH := $(CUDA_PATH)/lib

# Remove function
RM = rm -f
 
# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS := -lcudadevrt -lcudart
NVCCFLAGS :=
GccFLAGS = -fopenmp 

debug: GccFLAGS += -DDEBUG -g -Wall
debug: MPIFLAGS += -DDEBUG
debug: NVCCFLAGS += -g -G
debug: all

# The build target executable:
TARGET = brightness

all: build

build: $(TARGET)

$(TARGET): lib/dlink.o lib/main.o lib/alg/$(TARGET).o lib/img/imghandler.o lib/alg/locationhandler.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -link

lib/dlink.o: lib/alg/$(TARGET).o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

lib/main.o: lib/main.cpp lib/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@

lib/alg/locationhandler.o: lib/alg/locationhandler.cpp lib/alg/locationhandler.h lib/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@
	
lib/alg/$(TARGET).o: lib/alg/$(TARGET).cu lib/config.h
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 

lib/img/imghandler.o: lib/img/imghandler.cpp lib/config.h
	$(GCC) $(GccFLAGS) -c $< -o $@
	
clean:
	$(RM) $(TARGET) *.o lib/*.o *.tar* *.core*

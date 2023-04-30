cc        := g++
name      := pro
workdir   := workspace
srcdir    := src
objdir    := objs
stdcpp    := c++14
cuda_home := /home/hrm/TensorRT_Lib/trt8cuda102cudnn8
syslib    := /home/hrm/TensorRT_Lib/lib
cpp_pkg   := /home/hrm/TensorRT_Lib/cpp-packages
cuda_arch := 
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc)

# Define the path and dependency mk file for cpp lookup
cpp_srcs := $(shell find $(srcdir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(srcdir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

# Define the path and dependency mk file for cu lookup
cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(srcdir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

# Define the library files needed for opencv and cuda
link_cuda      := cudart cudnn
link_trtpro    := 
link_tensorRT  := nvinfer nvinfer_plugin
link_opencv    := opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui
link_sys       := stdc++ dl protobuf
link_librarys  := $(link_cuda) $(link_tensorRT) $(link_sys) $(link_opencv)

# Define the path to the header file, please note that there should be no spaces after the slash
# Only the path needs to be written, not the -I
include_paths := src              \
    $(cuda_home)/include/cuda     \
	$(cuda_home)/include/tensorRT \
	$(cpp_pkg)/opencv4.1.1/include  \
	$(cuda_home)/include/protobuf

# Define the library file path, just write the path, no need to write -L
library_paths := $(cuda_home)/lib64 $(syslib) $(cpp_pkg)/opencv4.1.1/lib

# Splice the library path into a string, e.g. a b c => a:b:c
# Then make LD_LIBRARY_PATH=a:b:c
empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

# Splice library paths and header paths into one, batch automatically add -I, -L, -l
run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -fPIC -fopenmp -pthread
cu_compile_flags  := -std=$(stdcpp) -w -g -O0 $(cuda_arch) -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

# If the header file is modified, the directive here allows it to automatically compile the dependent cpp or cu file
ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

$(name)   : $(workdir)/$(name)

all       : $(name)
run       : $(name)
	@cd $(workdir) && ./$(name) $(run_args)

$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) $^ -o $@ $(link_flags)

$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

# Compile cpp dependencies and generate mk files
$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
    
# Compile the dependencies of the cu file and generate the cumk file
$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

# Define cleanup instructions
clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/*.trtmodel $(workdir)/*.onnx

# Prevent symbols from being treated as files
.PHONY : clean run $(name)

# Export the dependent library paths to make it work
export LD_LIBRARY_PATH:=$(library_path_export)
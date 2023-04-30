// Header files for compiling the model
#include <NvInfer.h>
// header file for onnx parser
#include <onnx-tensorrt-release-8.2-GA/NvOnnxParser.h>
// Runtime headers for inference
#include <NvInferRuntime.h>
// Initialize the plugin's headers
#include <NvInferPlugin.h>
// CUDA headers
#include <cuda_runtime.h>
// OpenCV headers
#include <opencv2/opencv.hpp>
// System headers
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <chrono>
//exit()
#include <stdlib.h>
//signal()
#include <signal.h>

using namespace std;


//Exception Determination Tool
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

//Log Printing Tool
inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}
class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
            // Printing colored characters:
            // Part of the color code: https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;


// Management of pointer parameters returned by nv via smart pointers
// Automatic memory release to avoid leaks
template<typename _T>
static shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

static bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}


// Load model file
vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}


//global variables
//-------------------------------------------------------------------------------------------------------------
// Define global variables for the flow control file
cudaStream_t stream = nullptr;

// Define an array of GPU file pointers and get global variables for GPU input and output indexes
void* buffers[2];// GPU memory pointer array
int inputIndex;
int outputIndex;

// Allocate global variables for input image memory
int input_batch = 1;
int input_channel = 3;
int input_height = 256;
int input_width = 256;
int input_numel = input_batch * input_channel * input_height * input_width;
float* input_data_host = nullptr;//CPU cache pointer

// global variable for allocating output image memory
int output_batch = 1;
int output_channel = 3;
int output_height = 256;
int output_width = 256;
int output_numel = output_batch * output_channel * output_height * output_width;
float* output_data_host = nullptr;//CPU cache pointer
//-------------------------------------------------------------------------------------------------------------



//Ctrl+C for freeing memory
void  Handler(int signo){
    cout<<"free buffer in Handler"<<endl;
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(buffers[inputIndex]));
    checkRuntime(cudaFree(buffers[outputIndex]));
    exit(0);
}

//inference
void inference(){

	float f, fps;
    float FPS[16];
    int i, Fcnt=0;
    int jpg_count = 1;
    string jpg_mame;
    for(i=0;i<16;i++) FPS[i]=0.0;
    chrono::steady_clock::time_point Tbegin,Tend;

    // Define the log object
    TRTLogger logger;
    // Initialize the plug-in
    initLibNvInferPlugins(&logger, "");

    //Get runtime
    nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(logger);
    //getengine

    //before compression
    //auto engine_data = load_file("./model_in_nvidia/model_in_cyclegan/lir2rgb_cyclegan/all_net_G_A.fp32.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_cyclegan/lir2rgb_cyclegan/all_net_G_A.fp16.trtmodel");

    //auto engine_data = load_file("./model_in_nvidia/model_in_cyclegan/rgbn2rgb_cyclegan/all_net_G_A.fp32.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_cyclegan/rgbn2rgb_cyclegan/all_net_G_A.fp16.trtmodel");

    //auto engine_data = load_file("./model_in_nvidia/model_in_cyclegan/pir2rgb_cyclegan/all_net_G_A.fp32.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_cyclegan/pir2rgb_cyclegan/all_net_G_A.fp16.trtmodel");

    // After compression
    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/lir2rgb_compress_cyclegan/fid_latest_net_G.fp32.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/lir2rgb_compress_cyclegan/fid_latest_net_G.fp16.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/lir2rgb_compress_cyclegan/mac_latest_net_G.fp32.trtmodel");
    auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/lir2rgb_compress_cyclegan/mac_latest_net_G.fp16.trtmodel");

    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/rgbn2rgb_compress_cyclegan/fid_latest_net_G.fp32.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/rgbn2rgb_compress_cyclegan/fid_latest_net_G.fp16.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/rgbn2rgb_compress_cyclegan/mac_latest_net_G.fp32.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/rgbn2rgb_compress_cyclegan/mac_latest_net_G.fp16.trtmodel");

    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/pir2rgb_compress_cyclegan/fid_latest_net_G.fp32.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/pir2rgb_compress_cyclegan/fid_latest_net_G.fp16.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/pir2rgb_compress_cyclegan/mac_latest_net_G.fp32.trtmodel");
    //auto engine_data = load_file("./model_in_nvidia/model_in_compress_cyclegan/pir2rgb_compress_cyclegan/mac_latest_net_G.fp16.trtmodel");

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        delete runtime;
        return;
    }
    //Get the context
    nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();

    // Global variable initialization
    //-------------------------------------------------------------------------------------------------------------
    // Define the flow control file
    checkRuntime(cudaStreamCreate(&stream));

    // Define the GPU file pointer array and get the GPU input and output indexes
    inputIndex = engine->getBindingIndex("input");
    outputIndex = engine->getBindingIndex("output");

    // Allocate input image memory
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));// Allocate memory
    checkRuntime(cudaMalloc(&buffers[inputIndex], input_numel * sizeof(float)));//Allocate GPU graphics memory

    // Allocate output image memory
    checkRuntime(cudaMallocHost(&output_data_host, output_numel * sizeof(float)));// Allocate memory
    checkRuntime(cudaMalloc(&buffers[outputIndex], output_numel * sizeof(float)));//Allocate GPU graphics memory
    //-------------------------------------------------------------------------------------------------------------

    // Judgment on the input data type
    if(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT){
        cout<<"input:FP32"<<endl;
    }else if(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kHALF){
        cout<<"input:FP16"<<endl;
    }else if(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kINT8){
        cout<<"input:INT8"<<endl;
    }else if(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kINT32){
        cout<<"input:INT32"<<endl;
    }else{
        cout<<"input:Unknow"<<endl;
    }
    // Judgment on the output data type
    if(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT){
        cout<<"output:FP32"<<endl;
    }else if(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kHALF){
        cout<<"output:FP16"<<endl;
    }else if(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kINT8){
        cout<<"output:INT8"<<endl;
    }else if(engine->getBindingDataType(outputIndex) == nvinfer1::DataType::kINT32){
        cout<<"output:INT32"<<endl;
    }else{
        cout<<"output:Unknow"<<endl;
    }


    while(1){

        // Timer start
        Tbegin = chrono::steady_clock::now();

        //Read image from file or camera
        cv::Mat image;
        jpg_mame = "./dataset_original/lir2rgb_test/testA/" + to_string(jpg_count) + ".jpg";
        //jpg_mame = "./dataset_original/rgbn2rgb_test/testA/" + to_string(jpg_count) + ".jpg";
        //jpg_mame = "./dataset_original/pir2rgb_test/testA/" + to_string(jpg_count) + ".jpg";
        cout<<jpg_mame <<endl;

        image = cv::imread(jpg_mame);

        if (image.empty()){
            cerr << "ERROR: Unable to grab from the image" << endl;
            break;
        }

        cv::imshow("input", image);

        //Mat->Tensor
        float mean[] = {0.406, 0.456, 0.485};
        float std[]  = {0.225, 0.224, 0.229};
        cv::resize(image, image, cv::Size(input_width, input_height));
        int image_area = image.cols * image.rows;
        unsigned char* pimage = image.data;
        float* phost_r = input_data_host + image_area * 0;
        float* phost_g = input_data_host + image_area * 1;
        float* phost_b = input_data_host + image_area * 2;
        for(int i = 0; i < image_area; ++i, pimage += 3){
            
            *phost_b++ = (pimage[0] / 255.0f - mean[0]) / std[0];
            *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
            *phost_r++ = (pimage[2] / 255.0f - mean[2]) / std[2];
        }

        //CPU->GPU
        checkRuntime(cudaMemcpyAsync(buffers[inputIndex], input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

        //Begin inference model
        bool success = execution_context->enqueueV2(buffers, stream, nullptr);
        if(success== true){
            cout<<"convert success"<<endl;
        }

        //GPU->CPU
        checkRuntime(cudaMemcpyAsync(output_data_host, buffers[outputIndex], output_numel * sizeof(float), cudaMemcpyDeviceToHost, stream));

        //Wait for inference copy etc. to finish
        checkRuntime(cudaStreamSynchronize(stream));

        //Tensor-> Mat
        cv::Mat image_output(output_width, output_height, CV_32FC3);
        for(int c=0; c<output_channel; c++){
            for(int h=0; h<output_height; h++){
                for(int w=0; w<output_width; w++){
                    image_output.ptr<float>(h)[image_output.channels() * w + c] = output_data_host[ c*output_width*output_height + h*output_width + w];
                }
            }
        }
        cv::add(image_output,1.0,image_output);
        cv::divide(image_output,2.0,image_output);
        image_output.convertTo(image_output,CV_8U,255);
        cv::cvtColor(image_output,image_output,cv::COLOR_RGB2BGR);

        // Timer stop
        Tend = chrono::steady_clock::now();

        // Time consuming
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        cout << "every process consume:"<< f << "ms" <<endl;

        // Calculate fps
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        fps = f /16;
        cout <<"fps:"<< fps << endl;

        // Save Image

        // Before compression
        //cv::imwrite("./dataset_original/lir2rgb_test/cyclegan_fp32/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/lir2rgb_test/cyclegan_fp16/" + to_string(jpg_count) + ".jpg", image_output);

        //cv::imwrite("./dataset_original/rgbn2rgb_test/cyclegan_fp32/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/rgbn2rgb_test/cyclegan_fp16/" + to_string(jpg_count) + ".jpg", image_output);

        //cv::imwrite("./dataset_original/pir2rgb_test/cyclegan_fp32/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/pir2rgb_test/cyclegan_fp16/" + to_string(jpg_count) + ".jpg", image_output);

        // After compression
        //cv::imwrite("./dataset_original/lir2rgb_test/cyclegan_compress_fid_fp32/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/lir2rgb_test/cyclegan_compress_fid_fp16/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/lir2rgb_test/cyclegan_compress_mac_fp32/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/lir2rgb_test/cyclegan_compress_mac_fp16/" + to_string(jpg_count) + ".jpg", image_output);

        //cv::imwrite("./dataset_original/rgbn2rgb_test/cyclegan_compress_fid_fp32/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/rgbn2rgb_test/cyclegan_compress_fid_fp16/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/rgbn2rgb_test/cyclegan_compress_mac_fp32/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/rgbn2rgb_test/cyclegan_compress_mac_fp16/" + to_string(jpg_count) + ".jpg", image_output);

        //cv::imwrite("./dataset_original/pir2rgb_test/cyclegan_compress_fid_fp32/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/pir2rgb_test/cyclegan_compress_fid_fp16/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/pir2rgb_test/cyclegan_compress_mac_fp32/" + to_string(jpg_count) + ".jpg", image_output);
        //cv::imwrite("./dataset_original/pir2rgb_test/cyclegan_compress_mac_fp16/" + to_string(jpg_count) + ".jpg", image_output);


        // Show images
        cv::putText(image_output, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(0, 0, 255));
        cv::imshow("output",image_output);

        // ready to exit when esc key is pressed is detected
        char esc = cv::waitKey(1);
        if(esc == 27){
            // Destroy all windows
            cv::destroyAllWindows();
            // Free memory
            cout<<"free buffer in esc"<<endl;
            checkRuntime(cudaStreamDestroy(stream));
            checkRuntime(cudaFreeHost(input_data_host));
            checkRuntime(cudaFreeHost(output_data_host));
            checkRuntime(cudaFree(buffers[inputIndex]));
            checkRuntime(cudaFree(buffers[outputIndex]));

            exit(0);
            break;
        }

        // Continuous detection
        jpg_count ++;
        if(jpg_count > 4320){
        //if(jpg_count > 5128){
        //if(jpg_count > 2364){

            // Destroy all windows
            cv::destroyAllWindows();
            // Free memory
            cout<<"free buffer in esc"<<endl;
            checkRuntime(cudaStreamDestroy(stream));
            checkRuntime(cudaFreeHost(input_data_host));
            checkRuntime(cudaFreeHost(output_data_host));
            checkRuntime(cudaFree(buffers[inputIndex]));
            checkRuntime(cudaFree(buffers[outputIndex]));

            exit(0);
            break;
        }
    }

    // Destroy all windows
    cv::destroyAllWindows();

    // Free memory
    cout<<"free buffer in inference"<<endl;
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(buffers[inputIndex]));
    checkRuntime(cudaFree(buffers[outputIndex]));
}

int main(){

    // Handling Ctrl+C for freeing memory
    signal(SIGINT, Handler);

    std::cout << "OpenCV Version:" << CV_VERSION << std::endl;
    std::cout << cv::getBuildInformation() << std::endl;

    inference();

    return 0;
}

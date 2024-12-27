#include "acl/acl.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <map>


using namespace std;
int32_t deviceId_ = 0;
uint32_t modelId;

uint32_t data0Size;
uint32_t data1Size;
void *data0HostData;
void *data0DeviceData;
void *data1HostData;
void *data1DeviceData;
aclmdlDataset *inputDataSet;
aclDataBuffer *inputDataBuffer0;
aclDataBuffer *inputDataBuffer1;
aclmdlDataset *outputDataSet;
aclDataBuffer *outputDataBuffer;
aclmdlDesc *modelDesc;
size_t outputDataSize = 0;
void *outputDeviceData;
void *outputHostData;


// AscendCL初始化、运行管理资源申请（指定计算设备）
void InitResource()
{
	aclError ret = aclInit(nullptr);
	ret = aclrtSetDevice(deviceId_);
}


// 申请内存，使用C/C++标准库的函数将测试图片读入内存
void ReadBin(const char *picturePath, void *&inputBuff, uint32_t &fileSize)
{
	string fileName = picturePath;
	ifstream binFile(fileName, ifstream::binary);
	binFile.seekg(0, binFile.end);
	fileSize = binFile.tellg();
	binFile.seekg(0, binFile.beg);
	aclError ret = aclrtMallocHost(&inputBuff, fileSize);
	binFile.read((char*)inputBuff, fileSize);
	binFile.close();
}


// 申请Device侧的内存，再以复制内存的方式将内存中的图片数据传输到Device
void CopyDataFromHostToDevice()
{
	aclError ret = aclrtMalloc(&data0DeviceData, data0Size, ACL_MEM_MALLOC_HUGE_FIRST);
	ret = aclrtMemcpy(data0DeviceData, data0Size, data0HostData, data0Size, ACL_MEMCPY_HOST_TO_DEVICE);
	
	ret = aclrtMalloc(&data1DeviceData, data1Size, ACL_MEM_MALLOC_HUGE_FIRST);
	ret = aclrtMemcpy(data1DeviceData, data1Size, data1HostData, data1Size, ACL_MEMCPY_HOST_TO_DEVICE);
}


// 准备模型推理的输入数据结构
void CreateModelInput()
{
	// 创建aclmdlDataset类型的数据，描述模型推理的输入
	inputDataSet = aclmdlCreateDataset();
	
	inputDataBuffer0 = aclCreateDataBuffer(data0DeviceData, data0Size);
	aclError ret = aclmdlAddDatasetBuffer(inputDataSet, inputDataBuffer0);
	
	inputDataBuffer1 = aclCreateDataBuffer(data1DeviceData, data1Size);
	ret = aclmdlAddDatasetBuffer(inputDataSet, inputDataBuffer1);
}


// 准备模型推理的输出数据结构
void CreateModelOutput()
{
	// 创建模型描述信息
	modelDesc =  aclmdlCreateDesc();
	aclError ret = aclmdlGetDesc(modelDesc, modelId);
	
	// 创建aclmdlDataset类型的数据，描述模型推理的输出
	outputDataSet = aclmdlCreateDataset();
	
	// 获取模型输出数据需占用的内存大小，单位为Byte
	outputDataSize = aclmdlGetOutputSizeByIndex(modelDesc, 0);
	
	// 申请输出内存
	ret = aclrtMalloc(&outputDeviceData, outputDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
	outputDataBuffer = aclCreateDataBuffer(outputDeviceData, outputDataSize);
	ret = aclmdlAddDatasetBuffer(outputDataSet, outputDataBuffer);
}


// 将图片数据读入内存
void LoadPicture(const char* data0path, const char* data1path)
{
	ReadBin(data0path, data0HostData, data0Size);
	ReadBin(data1path, data1HostData, data1Size);
	CopyDataFromHostToDevice();
}


// 加载模型
void LoadModel(const char* modelPath)
{
	aclError ret = aclmdlLoadFromFile(modelPath, &modelId);
	if (ret != ACL_SUCCESS) {
        printf("load model from file failed, model file is %s, errorCode is %d",
                  modelPath, static_cast<int32_t>(ret));
        }
	printf("model id is %d\n", modelId);
}


// 执行推理
void Inference()
{
    CreateModelInput();
    CreateModelOutput();
    aclError ret = aclmdlExecute(modelId, inputDataSet, outputDataSet);
}


// 在终端上屏显测试图片的top5置信度的类别编号
void PrintResult()
{
	aclError ret = aclrtMallocHost(&outputHostData, outputDataSize);
	ret = aclrtMemcpy(outputHostData, outputDataSize, outputDeviceData, outputDataSize, ACL_MEMCPY_DEVICE_TO_HOST);
	float* outFloatData = reinterpret_cast<float *>(outputHostData);
	
	map<float, unsigned int, greater<float>> resultMap;
	for (unsigned int j = 0; j < 768;++j)
	{
		resultMap[*outFloatData] = j;
		outFloatData++;
		printf("%d:%lf\n", j, *outFloatData);
	}
	
	int cnt = 0;
	for (auto it = resultMap.begin();it != resultMap.end();++it)
	{
		if(++cnt > 10)
		{
			break;
		}
		printf("top %d: index[%d] value[%lf] \n", cnt, it->second, it->first);
	}
}


// 卸载模型
void UnloadModel()
{
	aclmdlDestroyDesc(modelDesc);
	aclmdlUnload(modelId);
}


// 释放内存、销毁推理相关的数据类型，防止内存泄露
void UnloadPicture()
{
	aclError ret = aclrtFreeHost(data0HostData);
	data0HostData = nullptr;
	ret = aclrtFree(data0DeviceData);
	data0DeviceData = nullptr;
	aclDestroyDataBuffer(inputDataBuffer0);
	inputDataBuffer0 = nullptr;
	aclmdlDestroyDataset(inputDataSet);
	inputDataSet = nullptr;
	
	ret = aclrtFreeHost(outputHostData);
	outputHostData = nullptr;
	ret = aclrtFree(outputDeviceData);
	outputDeviceData = nullptr;
	aclDestroyDataBuffer(outputDataBuffer);
	outputDataBuffer = nullptr;
	aclmdlDestroyDataset(outputDataSet);
	outputDataSet = nullptr;
}


// AscendCL去初始化、运行管理资源释放（指定计算设备）
void DestroyResource()
{
	aclError ret = aclrtResetDevice(deviceId_);
	aclFinalize();
}

int main()
{
	// 1.定义一个资源初始化的函数，用于AscendCL初始化、运行管理资源申请（指定计算设备）
	InitResource();
	
	printf("resource init successed\n");
	// 2.定义一个模型加载的函数，加载图片分类的模型，用于后续推理使用
	const char *mdoelPath = "../model/bce_embedding_bs1_10.om";
	LoadModel(mdoelPath);
	printf("load model successed\n");
	
	// 3.定义一个读图片数据的函数，将测试图片数据读入内存，并传输到Device侧，用于后续推理使用
	const char *data0path = "../data/0.bin";
	const char *data1path = "../data/1.bin";
	
	LoadPicture(data0path, data1path);
	printf("read pictures successed\n");
	
	// 4.定义一个推理的函数，用于执行推理
	Inference();
	printf("infer successed\n");
	
	// 5.定义一个推理结果数据处理的函数，用于在终端上屏显测试图片的top5置信度的类别编号
	PrintResult();
	
	// 6.定义一个模型卸载的函数，卸载图片分类的模型
	UnloadModel();
	
	// 7.定义一个函数，用于释放内存、销毁推理相关的数据类型，防止内存泄露
	// UnloadPicture();
	
	// 8.定义一个资源去初始化的函数，用于AscendCL去初始化、运行管理资源释放（指定计算设备）
	DestroyResource();
}

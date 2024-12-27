/*
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <iostream>
#include <fstream>
#include <map>
#include <math.h>
#include <sstream>
#include <algorithm>
#include <functional>
#include "utils.h"
#include "model_process.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "acl/acl.h"

using namespace std;
namespace {
    const int DIM_COUNT = 1;
    const int DIM_0 = 0;
    const int DIM_1 = 1;
    // const int DIM_2 = 2;
    // const int DIM_3 = 3;
}
ModelProcess::ModelProcess() : modelId_(0), modelWorkSize_(0), modelWeightSize_(0), modelWorkPtr_(nullptr),
                               modelWeightPtr_(nullptr), loadFlag_(false), modelDesc_(nullptr), input_(nullptr), output_(nullptr)
{
}

ModelProcess::~ModelProcess()
{
    UnloadModel();
    DestroyModelDesc();
    DestroyInput();
    DestroyOutput();
}
void ModelProcess::GetRunMode(aclrtRunMode runMode)
{
    isDevice_ = (runMode == ACL_DEVICE);
}

Result ModelProcess::LoadModel(const char *modelPath)
{
    if (loadFlag_) {
        ERROR_LOG("model has already been loaded");
        return FAILED;
    }
    aclError ret = aclmdlQuerySize(modelPath, &modelWorkSize_, &modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("query model failed, model file is %s, errorCode is %d",
                  modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }
    printf("modelWorkSize: %ld  modelWeightSize %ld \n", modelWorkSize_, modelWeightSize_);
    modelWorkSize_ = 1024 * 1024 * 1024;
    
    /*
    ret = aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc buffer for work failed, require size is %zu, errorCode is %d",
                  modelWorkSize_, static_cast<int32_t>(ret));
        return FAILED;
    }
    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("malloc buffer for weight failed, require size is %zu, errorCode is %d",
                  modelWeightSize_, static_cast<int32_t>(ret));
        return FAILED;
    }
    ret = aclmdlLoadFromFileWithMem(modelPath, &modelId_, modelWorkPtr_,
                                    modelWorkSize_, modelWeightPtr_, modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("load model from file failed, model file is %s, errorCode is %d",
                  modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }
    */
    
    
    ret = aclmdlLoadFromFile(modelPath, &modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("load model from file failed, model file is %s, errorCode is %d",
                  modelPath, static_cast<int32_t>(ret));
        return FAILED;
    }
    loadFlag_ = true;
    INFO_LOG("load model %s success", modelPath);
    return SUCCESS;
}


size_t ModelProcess::GetInputNums()
{
    size_t inputNums = aclmdlGetNumInputs(modelDesc_);
    return inputNums;
}

Result ModelProcess::CreateModelDesc()
{
    modelDesc_ = aclmdlCreateDesc();
    if (modelDesc_ == nullptr) {
        ERROR_LOG("create model description failed");
        return FAILED;
    }

    aclError ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get model description failed, modelId is %u, errorCode is %d",
                  modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }

    INFO_LOG("create model description success");

    size_t numInputs = aclmdlGetNumInputs(modelDesc_);
    INFO_LOG("nums of inputs: %ld", numInputs);

    const char* inputName = aclmdlGetInputNameByIndex(modelDesc_, 0);
    INFO_LOG("input 1 nams is %s", inputName);

    inputName = aclmdlGetInputNameByIndex(modelDesc_, 1);
    INFO_LOG("input 2 names is %s", inputName);
    
    /*
    long unsigned int gearCount = 0;
    ret = aclmdlGetInputDynamicGearCount(modelDesc_, -1, &gearCount);

    printf("gearInput Size is %ln\n", gearCount);

    aclmdlIODims dims[gearCount];
    ret = aclmdlGetInputDynamicDims(modelDesc_, -1, dims, gearCount);

    printf("get max dims from model %d: %d\n", dims[0], dims[1]);
   */
   
    return SUCCESS;
}

void ModelProcess::DestroyModelDesc()
{
    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }
    INFO_LOG("destroy model description success");
}


Result ModelProcess::SetTensorDesc(int idx, int dims0, int dims1)
{
// ......
// 创建Tensor描述信息
// shape需要和给定的输入数据的shape一致
  int64_t shapes[] = {dims0, dims1};
  aclTensorDesc *inputDesc = aclCreateTensorDesc(ACL_INT64, 2, shapes, ACL_FORMAT_ND);
  // 设置index为0的动态输入的Tensor描述信息
  aclError ret = aclmdlSetDatasetTensorDesc(input_, inputDesc, idx);
  // ret = aclmdlSetDatasetTensorDesc(input_, inputDesc, 1);
  
  return SUCCESS;
}

Result ModelProcess::ModelSetDynamicInfo(int dims0, int dims1)
{
    size_t index;
    aclError ret = aclmdlGetInputIndexByName(modelDesc_, "input_ids", &index);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("get input index by name[%s] failed, errorCode = %d.",
                  ACL_DYNAMIC_TENSOR_NAME, static_cast<int32_t>(ret));
        return FAILED;
    }

    currentDims_.dimCount = DIM_COUNT;
    currentDims_.dims[DIM_0] = dims0;
    // currentDims_.dims[DIM_1] = dims1;
    // currentDims_.dims[DIM_2] = dims2;
    // currentDims_.dims[DIM_3] = dims3;
    // ret = aclmdlSetInputDynamicDims(modelId_, input_, index, &currentDims_);
    size_t batchSize = 1;
    ret = aclmdlSetDynamicBatchSize(modelId_, input_, index, batchSize);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("set dynamic dims failed, errorCode = %d.", static_cast<int32_t>(ret));
        return FAILED;
    }
    return SUCCESS;
}
Result ModelProcess::Execute()
{
    aclError ret = aclmdlExecute(modelId_, input_, output_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("execute model failed, modelId is %u, errorCode is %d",
                  modelId_, static_cast<int32_t>(ret));
        return FAILED;
    }

    INFO_LOG("model execute success");
    return SUCCESS;
}


Result ModelProcess::CreateInput(std::vector<std::pair<void*, unsigned int> >& imageBufs)
{
    uint32_t dataNum = aclmdlGetNumInputs(modelDesc_);
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }
    input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }
    
    void *inputDataBuffer;
    size_t bufferSize;
    
    for (size_t idx =0; idx < imageBufs.size(); idx++){
      inputDataBuffer = imageBufs[idx].first;
      bufferSize = imageBufs[idx].second;
      
      size_t inputSize = aclmdlGetInputSizeByIndex(modelDesc_, idx);
      
      aclmdlIODims *dims;
      aclError err = aclmdlGetInputDims(modelDesc_, idx, dims);
      aclDataType inputType = aclmdlGetInputDataType(modelDesc_, idx);
      INFO_LOG("inputId is %ld, inputSize is %ld, bufferSize %ld inputType %d", idx, inputSize, bufferSize, inputType);
      for (int j=0;j<2;j++) printf("%ld ", dims->dims[j]);
      
    aclDataBuffer *inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }
    
    // void* inputDataHost = 
    // int64* inputtest = reinterpret_cast<int64 *>(inputDataBuffer);
    // for (int j=0;j<512;j++) printf(" %ld ", *(inputtest+j));
    
    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
        (void)aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return FAILED;
    }
    }
    
    /*
    size_t dynamicIdx = 0;
    aclError ret = aclmdlGetInputIndexByName(modelDesc_, ACL_DYNAMIC_TENSOR_NAME, &dynamicIdx);
    if ((ret == ACL_SUCCESS) && (dynamicIdx == (dataNum - 1))) {
        size_t dataLen = aclmdlGetInputSizeByIndex(modelDesc_, dynamicIdx);
        void *data = nullptr;
        ret = aclrtMalloc(&data, dataLen, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("malloc device memory failed, errorCode = %d.", static_cast<int32_t>(ret));
            return FAILED;
        }
        aclDataBuffer *dataBuf = aclCreateDataBuffer(data, dataLen);
        if (dataBuf == nullptr) {
            ERROR_LOG("Create data buffer error");
            return FAILED;
        }
        aclError ret = aclmdlAddDatasetBuffer(input_, dataBuf);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Add dataset buffer error %d", ret);
            ret = aclDestroyDataBuffer(dataBuf);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("Destroy dataset buffer error %d", ret);
            }
            dataBuf = nullptr;
            return FAILED;
        }
    }
    */

    INFO_LOG("create model input success");
    return SUCCESS;
}

Result ModelProcess::CreateInput(void *inputDataBuffer, size_t bufferSize)
{
    uint32_t dataNum = aclmdlGetNumInputs(modelDesc_);
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }
    if (input_ == nullptr || !input_) input_ = aclmdlCreateDataset();
    if (input_ == nullptr) {
        ERROR_LOG("can't create dataset, create input failed");
        return FAILED;
    }
    
    aclDataBuffer *inputData = aclCreateDataBuffer(inputDataBuffer, bufferSize);
    if (inputData == nullptr) {
        ERROR_LOG("can't create data buffer, create input failed");
        return FAILED;
    }
    aclError ret = aclmdlAddDatasetBuffer(input_, inputData);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
        (void)aclDestroyDataBuffer(inputData);
        inputData = nullptr;
        return FAILED;
    }
    
    
    /*
    size_t dynamicIdx = 1;
    ret = aclmdlGetInputIndexByName(modelDesc_, ACL_DYNAMIC_TENSOR_NAME, &dynamicIdx);
    printf("dynamic idx: %d\n", dynamicIdx);
    if ((ret == ACL_SUCCESS) && (dynamicIdx == (dataNum - 1))) {
        size_t dataLen = aclmdlGetInputSizeByIndex(modelDesc_, dynamicIdx);
        void *data = nullptr;
        ret = aclrtMalloc(&data, dataLen, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("malloc device memory failed, errorCode = %d.", static_cast<int32_t>(ret));
            return FAILED;
        }
        aclDataBuffer *dataBuf = aclCreateDataBuffer(data, dataLen);
        if (dataBuf == nullptr) {
            ERROR_LOG("Create data buffer error");
            return FAILED;
        }
        aclError ret = aclmdlAddDatasetBuffer(input_, dataBuf);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Add dataset buffer error %d", ret);
            ret = aclDestroyDataBuffer(dataBuf);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("Destroy dataset buffer error %d", ret);
            }
            dataBuf = nullptr;
            return FAILED;
        }
    }
    */

    INFO_LOG("create model input success");
    return SUCCESS;
}
Result ModelProcess::GetInputSizeByIndex(const size_t index, size_t &inputSize)
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create input failed");
        return FAILED;
    }
    inputSize = aclmdlGetInputSizeByIndex(modelDesc_, index);
    return SUCCESS;
}
void ModelProcess::DestroyInput()
{
    if (input_ == nullptr || !input_) {
        return;
    }
    
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
        printf("start destroy %ld\n", i);
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
        (void)aclDestroyDataBuffer(dataBuffer);
    }
    (void)aclmdlDestroyDataset(input_);
    input_ = nullptr;
    INFO_LOG("destroy model input success");
}

Result ModelProcess::CreateOutput()
{
    if (modelDesc_ == nullptr) {
        ERROR_LOG("no model description, create ouput failed");
        return FAILED;
    }

    output_ = aclmdlCreateDataset();
    if (output_ == nullptr) {
        ERROR_LOG("can't create dataset, create output failed");
        return FAILED;
    }

    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
    
        const char *outputName = aclmdlGetOutputNameByIndex(modelDesc_, i);
        size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        
        aclmdlIODims dims[] = {0, 0};
        aclError err = aclmdlGetOutputDims(modelDesc_, i, dims);
        printf("输出为%s, 输出文件大小%lu \n", outputName, modelOutputSize);
        
        for (int j=0;j<2;j++){ printf("%ld ", dims->dims[j]);}
        
        
        void *outputBuffer = nullptr;
        modelOutputSize = 32*768*4;
        // modelOutputSize = 3*1*4;
        aclError ret = aclrtMalloc(&outputBuffer, modelOutputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't malloc buffer, size is %zu, create output failed, errorCode is %d",
                      modelOutputSize, static_cast<int32_t>(ret));
            return FAILED;
        }
        
        ret = aclrtMemset(outputBuffer, modelOutputSize, 0, modelOutputSize);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("aclrtMemset failed");
        }

        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, modelOutputSize);
        if (outputData == nullptr) {
            ERROR_LOG("can't create data buffer, create output failed");
            (void)aclrtFree(outputBuffer);
            return FAILED;
        }
        
        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("can't add data buffer, create output failed, errorCode is %d",
                      static_cast<int32_t>(ret));
            (void)aclrtFree(outputBuffer);
            (void)aclDestroyDataBuffer(outputData);
            return FAILED;
        }
    }
    INFO_LOG("create model output success");

    return SUCCESS;
}

void ModelProcess::OutputModelResult(std::vector<float>& outputs, int batch, int classNum)
{
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void *data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);
        
        printf("获取输出数据的长度%u \n", len);

        void *outHostData = nullptr;
        aclError ret = ACL_SUCCESS;
        float *outData = nullptr;
        if (!isDevice_) {
            aclError ret = aclrtMallocHost(&outHostData, len);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtMallocHost failed, malloc len[%u], errorCode[%d]",
                          len, static_cast<int32_t>(ret));
                return;
            }
            
            ret = aclrtMemset(outHostData, len, 0, len);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtMemset failed, memset len[%u], errorCode[%d]",
                          len, static_cast<int32_t>(ret));
                return;
            }

            // if app is running in host, need copy model output data from device to host
            ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtMemcpy failed, errorCode[%d]", static_cast<int32_t>(ret));
                (void)aclrtFreeHost(outHostData);
                return;
            }
            INFO_LOG("aclrtMallocHost--------success------------");
            outData = reinterpret_cast<float *>(outHostData);
            
            // for (int j = 0; j < len / sizeof(float) ; ++j) printf("%f ", *((float*)outHostData+j));
        } else {
            outData = reinterpret_cast<float *>(data);
        }
        
        
        // int classNum = 768;
        // int classNum = 3;
        for (int i = 0; i < batch; i++) {
            for (unsigned int j = 0; j < classNum; ++j) {
                printf("%f ", *(outData + i* classNum + j));
                outputs.push_back(*(outData + i* classNum + j));
            }
        }
        if (!isDevice_) {
            ret = aclrtFreeHost(outHostData);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtFreeHost failed, errorCode[%d]", static_cast<int32_t>(ret));
                return;
            }
        }
    }

    INFO_LOG("output data success");
    return;
}

void ModelProcess::OutputModelResultSoftMax(int classNum, int batchSize)
{
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void *data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);

        void *outHostData = nullptr;
        aclError ret = ACL_SUCCESS;
        float *outData = nullptr;
        if (!isDevice_) {
            aclError ret = aclrtMallocHost(&outHostData, len);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtMallocHost failed, malloc len[%u], errorCode[%d]",
                          len, static_cast<int32_t>(ret));
                return;
            }

            ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtMemcpy failed, errorCode[%d]", static_cast<int32_t>(ret));
                (void)aclrtFreeHost(outHostData);
                return;
            }
            outData = reinterpret_cast<float *>(outHostData);
        } else {
            outData = reinterpret_cast<float *>(data);
        }

        double total = 0.0;

        for (size_t j = 0; j < batchSize; j++)
        {
            for(int i = j*classNum; i < (j+1)*classNum; i++) {
                total += exp(outData[i]);
            }
            for(int i = j*classNum; i < (j+1)*classNum; i++) {
                outData[i] = exp(outData[i]) / total;
            }
            total = 0.0;
        }

        for (int i = 0; i < (len / sizeof(float) / classNum); i++) {
            map<float, unsigned int, greater<float>> resultMap;
            for (unsigned int j = 0; j < classNum; ++j) {
                resultMap[*outData] = j;
                outData++;
            }
            INFO_LOG("seq = %d---------------------", i);
            int cnt = 0;
            for (auto it = resultMap.begin(); it != resultMap.end(); ++it) {
                // print top 5
                if (++cnt > 5) {
                    break;
                }
                INFO_LOG("top %d: index[%d] value[%lf] cnt= %d", cnt, it->second, it->first, cnt);
            }
        }
        if (!isDevice_) {
            ret = aclrtFreeHost(outHostData);
            if (ret != ACL_SUCCESS) {
                ERROR_LOG("aclrtFreeHost failed, errorCode[%d]", static_cast<int32_t>(ret));
                return;
            }
        }
    }

    INFO_LOG("output data success");
    return;
}

void ModelProcess::DestroyOutput()
{
    if (output_ == nullptr) {
        return;
    }

    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
        aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void *data = aclGetDataBufferAddr(dataBuffer);
        (void)aclrtFree(data);
        (void)aclDestroyDataBuffer(dataBuffer);
    }

    (void)aclmdlDestroyDataset(output_);
    output_ = nullptr;
    INFO_LOG("destroy model output success");
}

void ModelProcess::UnloadModel()
{
    if (!loadFlag_) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    aclError ret = aclmdlUnload(modelId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("unload model failed, modelId is %u, errorCode is %d",
                  modelId_, static_cast<int32_t>(ret));
    }

    if (modelDesc_ != nullptr) {
        (void)aclmdlDestroyDesc(modelDesc_);
        modelDesc_ = nullptr;
    }

    if (modelWorkPtr_ != nullptr) {
        (void)aclrtFree(modelWorkPtr_);
        modelWorkPtr_ = nullptr;
        modelWorkSize_ = 0;
    }

    if (modelWeightPtr_ != nullptr) {
        (void)aclrtFree(modelWeightPtr_);
        modelWeightPtr_ = nullptr;
        modelWeightSize_ = 0;
    }

    loadFlag_ = false;
    INFO_LOG("unload model success, modelId is %u", modelId_);
    modelId_ = 0;
}

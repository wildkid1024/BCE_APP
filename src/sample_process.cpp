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
#include <vector>
#include <memory>
#include <unistd.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include "sample_process.h"
#include "acl/acl.h"
#include "utils.h"

using namespace std;
namespace {
    string g_testFile = "../data/";
    // const char *g_omModelPath = "../model/bce_embedding_bs1-32_1-512_linux_aarch64.om";
    // const string g_bpmPath = "/home/HwHiAiUser/Public/Models/bce-embedding-base_v1/sentencepiece.bpe.model";
    
    const char *g_omModelPath = "../model/bce-rerank/bce_rerank_bs1-32_1-512_linux_aarch64.om";
    const string g_bpmPath = "/home/HwHiAiUser/Public/Models/bce-reranker-base_v1/sentencepiece.bpe.model";
    int g_classNum = 768;
    int g_batchSize = 1;
    int g_seqlen = 512;
    // int g_modelWidth224 = 224;
    // int g_modelHeight224 = 224;
    // int g_modelWidth200 = 200;
    // int g_modelHeight200 = 200;
}

SampleProcess::SampleProcess() : deviceId_(0), context_(nullptr), stream_(nullptr)
{
}

SampleProcess::~SampleProcess()
{
    DestroyResource();
}

Result SampleProcess::InitResource(std::string &g_omModelPath, std::string &g_bpmPath)
{
    // ACL init
    const char *aclConfigPath = "./src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl init failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("acl init success");

    // set device
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl set device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("set device %d success", deviceId_);

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl create context failed, deviceId = %d, errorCode = %d",
                  deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl create stream failed, deviceId = %d, errorCode = %d",
                  deviceId_, static_cast<int32_t>(ret));
        return FAILED;
    }
    INFO_LOG("create stream success");

    // get run mode
    ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("acl get run mode failed, errorCode = %d", static_cast<int32_t>(ret));
        return FAILED;
    }
    isDevice_ = (runMode_ == ACL_DEVICE);
    modelProcess_.GetRunMode(runMode_);
    INFO_LOG("get run mode success");

    // model init
    Result modelRet = modelProcess_.LoadModel(g_omModelPath.c_str());
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModel failed");
        return FAILED;
    }

    modelRet = modelProcess_.CreateModelDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateModelDesc failed");
        return FAILED;
    }

    modelRet = modelProcess_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed");
        return FAILED;
    }
    
    tokenizer_.Load(g_bpmPath);
    INFO_LOG("load bpm success");
    return SUCCESS;
}


Result SampleProcess::InitStream(){
    aclError ret;
    aclError err = aclrtSetCurrentContext(context_);
    if (err != ACL_SUCCESS) {
    	ERROR_LOG("acl set context failed, errorCode = %d", err);
		return FAILED;
	}
}


Result SampleProcess::Process()
{
    
    // void *inputBuff[2] = {nullptr, nullptr};
    int modelSeqlen = 0;
    // std::vector<std::pair<void*, size_t>> inputBuffers;
    // std::vector<std::pair<void*, uint32_t>> imagesBuf;
    
    aclError ret;

    // int modelWidth = 0;
    // get model inputNums
    size_t inputNums = modelProcess_.GetInputNums();
    
    /*
    for (size_t inputIndex=0;inputIndex<inputNums;inputIndex++){
    ret = modelProcess_.GetInputSizeByIndex(inputIndex, devBufferSize);
    if (ret != SUCCESS) {
        ERROR_LOG("execute GetInputSizeByIndex failed");
        return FAILED;
    }
    if (!isDevice_) {
        ret = aclrtMallocHost(&inputBuff, devBufferSize);
        if (inputBuff == nullptr) {
            ERROR_LOG("aclrtMallocHost inputBuff failed, errorCode = %d.", static_cast < int32_t > (ret));
            return FAILED;
        }
    } else {
        aclError ret = aclrtMalloc(&inputBuff, devBufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("aclrtMalloc device buffer failed. size is %zu, errorCode is %d", devBufferSize,
                static_cast<int32_t>(ret));
            return FAILED;
        }
    }
     inputBuffers.emplace_back(std::make_pair(inputBuff, devBufferSize));
    }
    */

    

    for (size_t index = 0; index < g_batchSize; ++index) {
        if (index == 0) {
            modelSeqlen = g_seqlen;
            INFO_LOG("ModelSetDynamicInfo g_batchSize:%d, g_channels:%d",
                     g_batchSize, g_seqlen);
        }
        for (size_t idx=0;idx<inputNums;idx++){
            //inputBuff = inputBuffers[idx].first;
            // devBufferSize = inputBuffers[idx].second;
            
        void *inputBuff = nullptr;
        size_t devBufferSize;
        
        ret = modelProcess_.GetInputSizeByIndex(idx, devBufferSize);
		if (ret != SUCCESS) {
		    ERROR_LOG("execute GetInputSizeByIndex failed");
		    return FAILED;
		}
		if (!isDevice_) {
		    ret = aclrtMallocHost(&inputBuff, devBufferSize);
		    if (inputBuff == nullptr) {
		        ERROR_LOG("aclrtMallocHost inputBuff failed, errorCode = %d.", static_cast < int32_t > (ret));
		        return FAILED;
		    }
		} else {
		    aclError ret = aclrtMalloc(&inputBuff, devBufferSize, ACL_MEM_MALLOC_HUGE_FIRST);
		    if (ret != ACL_SUCCESS) {
		        ERROR_LOG("aclrtMalloc device buffer failed. size is %zu, errorCode is %d", devBufferSize,
		            static_cast<int32_t>(ret));
		        return FAILED;
		    }
		}
        
        std::string filePath = g_testFile + to_string(idx) + ".bin";
        printf("reading the file at path: %s\n", filePath.c_str());
        
        uint32_t oneBatchFileSize = ReadOneInput(filePath, inputBuff);
        if (oneBatchFileSize > devBufferSize) {
            ERROR_LOG("ReadOneBatch failed");
            return FAILED;
        }

        void *imageInfoBuf = Utils::MemcpyToDeviceBuffer(inputBuff, devBufferSize, runMode_);
        if (imageInfoBuf == nullptr) {
            ERROR_LOG("MemcpyToDeviceBuffer failed");
            return FAILED;
        }
        
        size_t size = devBufferSize;
        aclrtMemcpyKind policy = ACL_MEMCPY_DEVICE_TO_HOST;
        void *buffer = nullptr;
        aclError aclRet = aclrtMallocHost(&buffer, size);
        if ((aclRet != ACL_SUCCESS) || (buffer == nullptr)) {
            ERROR_LOG("Malloc memory failed, errorno:%d", aclRet);
        }
        ret = aclrtMemset(buffer, size, 0, size);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("aclrtMemset failed");
        }
        aclRet = aclrtMemcpy(buffer, size, imageInfoBuf, size, policy);
        // printf("file copy finished!\n");
        
        // int64* inputtest = reinterpret_cast<int64 *>(buffer);
        // for (int j=0;j<10;j++) printf(" %ld ", *(inputtest+j));

        
        ret = modelProcess_.CreateInput(imageInfoBuf, oneBatchFileSize);
        if (ret != SUCCESS) {
            ERROR_LOG("execute CreateInput failed");
            return FAILED;
        }
        
        ret = modelProcess_.SetTensorDesc(idx, g_batchSize, g_seqlen);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("SetTensorDesc  failed, errorCode is %d", static_cast<int32_t>(ret));
            return FAILED;
        }
        
        } // inputNums end

        ret = modelProcess_.Execute();
        if (ret != SUCCESS) {
            ERROR_LOG("execute inference failed");
            return FAILED;
        }
        // modelProcess_.DestroyInput();
        // modelProcess_.OutputModelResultSoftMax(g_classNum, g_batchSize);
        // modelProcess_.OutputModelResult();

    	// aclrtFree(inputBuff);
    	// modelProcess_.DestroyOutput();
    }
    return SUCCESS;
}

Result SampleProcess::Process(std::vector<std::string>& queries, std::vector<float>& outputs)
{
   
    int modelSeqlen = 0;
    aclError ret;
    
    size_t n = queries.size();
    size_t maxLen = 0;
   
    std::vector<int64_t> tmp;
    std::vector<vector<int64_t> > tmp_ids;
    for (auto query:queries){
    	tokenizer_.Encode(query, tmp);
    	tmp_ids.emplace_back(tmp);
    	
    	maxLen = max(tmp.size(), maxLen);
    	maxLen = min(maxLen, (size_t)512);
    	tmp.empty();
    }
    
    std::vector<int64_t> masks;
    std::vector<int64_t> ids;
    for (int i=0;i<n;i++){
    	size_t seqLen = min(tmp_ids[i].size(), (size_t)512);
    	for (int j=0;j<seqLen;j++){
    		ids.emplace_back(tmp_ids[i][j]);
    		masks.emplace_back(1);
    	}
    	for (int j=seqLen;j<maxLen;j++){
    		ids.emplace_back(1);
    		masks.emplace_back(0);
    	}
    }
    
    for (size_t i=0;i<ids.size();i++) printf("%ld ", ids[i]);
    
    size_t idsbyte = ids.size() * sizeof(int64_t);
    for (size_t i=0;i<1;i++){
        // size_t devBufferSize;   
        
        // ret = modelProcess_.GetInputSizeByIndex(0, devBufferSize);
        printf("vector size :%ld \n", idsbyte);
        void *idsInfoBuf = Utils::MemcpyToDeviceBuffer(ids.data(), idsbyte, runMode_);
        if (idsInfoBuf == nullptr) {
            ERROR_LOG("MemcpyToDeviceBuffer failed");
            return FAILED;
        }
        ret = modelProcess_.CreateInput(idsInfoBuf, idsbyte);
        if (ret != SUCCESS) {
            ERROR_LOG("execute CreateInput failed");
            return FAILED;
        }
        ret = modelProcess_.SetTensorDesc(0, n, maxLen);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("SetTensorDesc  failed, errorCode is %d", static_cast<int32_t>(ret));
            return FAILED;
        }
        
        // ret = modelProcess_.GetInputSizeByIndex(1, devBufferSize);
        void *maskInfoBuf = Utils::MemcpyToDeviceBuffer(masks.data(), idsbyte, runMode_);
        if (maskInfoBuf == nullptr) {
            ERROR_LOG("MemcpyToDeviceBuffer failed");
            return FAILED;
        }
        ret = modelProcess_.CreateInput(maskInfoBuf, idsbyte);
        if (ret != SUCCESS) {
            ERROR_LOG("execute CreateInput failed");
            return FAILED;
        }
        ret = modelProcess_.SetTensorDesc(1, n, maxLen);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("SetTensorDesc  failed, errorCode is %d", static_cast<int32_t>(ret));
            return FAILED;
        }
        
        }

        ret = modelProcess_.Execute();
        if (ret != SUCCESS) {
            ERROR_LOG("execute inference failed");
            return FAILED;
        }
        modelProcess_.DestroyInput();
        modelProcess_.OutputModelResult(outputs, n, 768);

    	// aclrtFree(inputBuff);
    	// modelProcess_.DestroyOutput();
    return SUCCESS;
}


Result SampleProcess::Process(std::string& query, std::vector<std::string>& answers, std::vector<float>& outputs)
{
   
    int modelSeqlen = 0;
    aclError ret;
    
    std::vector<int64_t> queryIds;
    tokenizer_.Encode(query, queryIds);
    size_t queryLen = queryIds.size();
    if (queryLen > 0) queryIds[0] = 2;
    for (auto& id:queryIds) printf("%d ", id);
    puts("");
    
    size_t numQuery = answers.size();
    size_t maxLen = 0;
    std::vector<std::vector<int64_t>> ids;
    
    for (size_t i=0; i<answers.size(); i++){
    	std::vector<int64_t> tmp;
    	tokenizer_.Encode(answers[i], tmp);
    	ids.push_back(tmp);
    	
    	maxLen = max(maxLen, tmp.size());
    }
    
    printf("maxLen is %d\n", maxLen);
    
    std::vector<int64_t> rowIds;
    std::vector<int64_t> masks;
    
    for (auto& tmp:ids){
        size_t len = tmp.size();
        size_t truncateLen = min(len, 512 - queryLen);
        maxLen = min(maxLen, 512 - queryLen);
        
        for (size_t j=0;j<truncateLen;j++){
        	rowIds.emplace_back(tmp[j]);
        	masks.push_back(1);
        }
        for (size_t j=0;j<queryLen;j++){
            rowIds.emplace_back(queryIds[j]);
        	masks.push_back(1);
        }
    	for (size_t j=truncateLen;j<maxLen;j++){
    		rowIds.emplace_back(1);
    		masks.push_back(0);
    	}
    	
    	for (auto& id:rowIds) printf("%d ", id);
    	puts("");
    }
    
    
    size_t idsbyte = rowIds.size() * sizeof(int64_t);
    for (size_t i=0;i<1;i++){
        // size_t devBufferSize;   
        
        // ret = modelProcess_.GetInputSizeByIndex(0, devBufferSize);
        printf("vector size :%ld \n", idsbyte);
        void *idsInfoBuf = Utils::MemcpyToDeviceBuffer(rowIds.data(), idsbyte, runMode_);
        if (idsInfoBuf == nullptr) {
            ERROR_LOG("MemcpyToDeviceBuffer failed");
            return FAILED;
        }
        ret = modelProcess_.CreateInput(idsInfoBuf, idsbyte);
        if (ret != SUCCESS) {
            ERROR_LOG("execute CreateInput failed");
            return FAILED;
        }
        ret = modelProcess_.SetTensorDesc(0, numQuery, maxLen+queryLen);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("SetTensorDesc  failed, errorCode is %d", static_cast<int32_t>(ret));
            return FAILED;
        }
        
        // ret = modelProcess_.GetInputSizeByIndex(1, devBufferSize);
        void *maskInfoBuf = Utils::MemcpyToDeviceBuffer(masks.data(), idsbyte, runMode_);
        if (maskInfoBuf == nullptr) {
            ERROR_LOG("MemcpyToDeviceBuffer failed");
            return FAILED;
        }
        ret = modelProcess_.CreateInput(maskInfoBuf, idsbyte);
        if (ret != SUCCESS) {
            ERROR_LOG("execute CreateInput failed");
            return FAILED;
        }
        ret = modelProcess_.SetTensorDesc(1, numQuery, maxLen+queryLen);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("SetTensorDesc  failed, errorCode is %d", static_cast<int32_t>(ret));
            return FAILED;
        }
        
        }

        ret = modelProcess_.Execute();
        if (ret != SUCCESS) {
            ERROR_LOG("execute inference failed");
            return FAILED;
        }
        modelProcess_.DestroyInput();
        modelProcess_.OutputModelResult(outputs, numQuery, 1);

    	// aclrtFree(inputBuff);
    	// modelProcess_.DestroyOutput();
    return SUCCESS;
}


// 需要写一份readBatch单个输入的函数

uint32_t SampleProcess::ReadOneInput(string filePath, void *&inputBuff)
{
    vector<string> fileVec{filePath};
    uint32_t fileSize = 0;
    uint32_t batchFileSize = 0;
    uint8_t i = 0;
    
    for (i; i < fileVec.size(); i++) {
        auto ret = Utils::ReadBinFile(fileVec[i], inputBuff, fileSize);
        if (ret != SUCCESS) {
            ERROR_LOG("read bin file failed, file name is %s", fileVec[i].c_str());
            return FAILED;
        }
        INFO_LOG("read bin file , file name is %s", fileVec[i].c_str());
        inputBuff = inputBuff + fileSize;
        batchFileSize = batchFileSize + fileSize;
  
        if (i == (fileVec.size() - 1)) {
            WARN_LOG("read bin file Num =%d.", i + 1);
            break;
        }
    }
    inputBuff = inputBuff - (i + 1) * fileSize;
    
    // int64* inputtest = reinterpret_cast<int64 *>(inputBuff);
    // for (int j=0;j<512;j++) printf(" %ld ", *(inputtest+j));
    INFO_LOG("ReadBinFile batchFileSize = %d", batchFileSize);
    return batchFileSize;
}


/*
uint32_t SampleProcess::ReadOneInput(string inputImageDir, void *&inputBuff)
{
    vector<string> fileVec;
    Utils::GetAllFiles(inputImageDir, fileVec);
    if (fileVec.empty()) {
        INFO_LOG("Failed to deal all empty path=%s.", inputImageDir.c_str());
        return FAILED;
    }
    uint32_t fileSize = 0;
    uint32_t batchFileSize = 0;
    uint8_t i = 0;
    
    for (i; i < fileVec.size(); i++) {
        auto ret = Utils::ReadBinFile(fileVec[i], inputBuff, fileSize);
        if (ret != SUCCESS) {
            ERROR_LOG("read bin file failed, file name is %s", fileVec[i].c_str());
            return FAILED;
        }
        INFO_LOG("read bin file , file name is %s", fileVec[i].c_str());
        inputBuff = inputBuff + fileSize;
        batchFileSize = batchFileSize + fileSize;
  
        if (i == (fileVec.size() - 1)) {
            WARN_LOG("read bin file Num =%d.", i + 1);
            break;
        }
    }
    inputBuff = inputBuff - (i + 1) * fileSize;
    INFO_LOG("ReadBinFile batchFileSize = %d", batchFileSize);
    return batchFileSize;
}
*/


uint32_t SampleProcess::ReadOneBatch(string inputImageDir, void *&inputBuff, int batchSize)
{
    vector<string> fileVec;
    Utils::GetAllFiles(inputImageDir, fileVec);
    if (fileVec.empty()) {
        INFO_LOG("Failed to deal all empty path=%s.", inputImageDir.c_str());
        return FAILED;
    }
    uint32_t fileSize = 0;
    uint32_t batchFileSize = 0;
    uint8_t i = 0;
    
    for (i; i < batchSize; i++) {
        auto ret = Utils::ReadBinFile(fileVec[i], inputBuff, fileSize);
        if (ret != SUCCESS) {
            ERROR_LOG("read bin file failed, file name is %s", fileVec[i].c_str());
            return FAILED;
        }
        INFO_LOG("read bin file , file name is %s", fileVec[i].c_str());
        inputBuff = inputBuff + fileSize;
        batchFileSize = batchFileSize + fileSize;

        if (i == (batchSize - 1)) {
            INFO_LOG("read bin file Num =%d.", i + 1);
            break;
        } else if (i == (fileVec.size() - 1)) {
            WARN_LOG("read bin file Num =%d.", i + 1);
            break;
        }
    }
    inputBuff = inputBuff - (i + 1) * fileSize;
    INFO_LOG("ReadBinFile batchFileSize = %d", batchFileSize);
    return batchFileSize;
}

uint32_t SampleProcess::ReadOneBatchPicHwc(string inputImageDir, void *&inputBuff, int batchSize, int seqlen)
{
    vector<string> fileVec;
    Utils::GetAllFiles(inputImageDir, fileVec);
    if (fileVec.empty()) {
        INFO_LOG("Failed to deal all empty path=%s.", inputImageDir.c_str());
        return FAILED;
    }
    uint32_t fileSize = 0;
    uint32_t batchFileSize = 0;
    uint32_t resizeWidth = 224;
    uint32_t resizeHeight = 224;

    for (uint8_t i = 0; i < batchSize; i++) {
        cv::Mat frame, reiszedImage, rsImageF32;
        frame=cv::imread(fileVec[i]);
        INFO_LOG("read pic file , file name is %s", fileVec[i].c_str());
        cv::resize(frame, reiszedImage, cv::Size(resizeWidth, resizeHeight));

        cv::Mat shipRGB;
        cv::cvtColor(reiszedImage, shipRGB, cv::COLOR_BGR2RGB);
        shipRGB.convertTo(rsImageF32, CV_32FC3);

        std::vector<cv::Mat> channels;
        cv::split(rsImageF32, channels);
        channels[2] -=123.0;
        channels[1] -=117.0;
        channels[0] -=104.0;
        channels[2] *=0.0142857142857143;
        channels[1] *=0.0142857142857143;
        channels[0] *=0.0142857142857143;

        cv::Mat result;
        cv::merge(channels, result);
        fileSize = ((resizeWidth) * (resizeHeight) * 3 * 4);
        memcpy(static_cast<uint8_t *>(inputBuff),  result.data, fileSize);

        inputBuff = inputBuff + fileSize;
        batchFileSize = batchFileSize + fileSize;
    }

    inputBuff = inputBuff - batchFileSize;
    INFO_LOG("Read Pic File batchFileSize = %d", batchFileSize);
    return batchFileSize;
}
void SampleProcess::DestroyResource()
{
    aclError ret;
    if (stream_ != nullptr) {
        ret = aclrtDestroyStream(stream_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("destroy stream failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        stream_ = nullptr;
    }
    INFO_LOG("end to destroy stream");

    if (context_ != nullptr) {
        ret = aclrtDestroyContext(context_);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("destroy context failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        context_ = nullptr;
    }
    INFO_LOG("end to destroy context");

    ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("reset device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
    }
    INFO_LOG("end to reset device %d", deviceId_);

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        ERROR_LOG("finalize acl failed, errorCode = %d", static_cast<int32_t>(ret));
    }
    INFO_LOG("end to finalize acl");
}

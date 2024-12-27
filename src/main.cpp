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
#include "sample_process.h"
#include "utils.h"

#include "httplib.h"
#include "json11.h"
using namespace std;


struct ChatSession {
    std::string history = "";
    std::string input = "";
    std::string output = "";
    int round = 0;
    int status = 0; // 0: 空闲 1: 结果生成好了 2: 已经写回了
};

std::map <std::string, ChatSession*> sessions;
std::mutex locker;

using Json = json11::Json;


struct WebConfig {
    std::string modelPath = "./model/bce-rerank/bce_rerank_bs1-32_1-512_linux_aarch64.om"; // 模型文件路径
    std::string sentenceModelPath = "/home/HwHiAiUser/Public/Models/bce-reranker-base_v1/sentencepiece.bpe.model"; // 词表模型文件的路径
    int port = 8090; // 端口号
};

void Usage() {
    std::cout << "Usage:" << std::endl;
    std::cout << "[-h|--help]:                  显示帮助" << std::endl;
    std::cout << "<-p|--path> <args>:           模型文件的路径" << std::endl;
    std::cout << "<-s|--spm> <args>:       词表模型文件的路径" << std::endl;
    std::cout << "<--port> <args>:              网页端口号" << std::endl;
}

void ParseArgs(int argc, char **argv, WebConfig &config) {
    std::vector <std::string> sargv;
    for (int i = 0; i < argc; i++) {
        sargv.push_back(std::string(argv[i]));
    }
    for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-h" || sargv[i] == "--help") {
            Usage();
            exit(0);
        } else if (sargv[i] == "-p" || sargv[i] == "--path") {
            config.modelPath = sargv[++i];
        } else if (sargv[i] == "-s" || sargv[i] == "--spm") {
            config.sentenceModelPath = sargv[++i];
        } else if (sargv[i] == "--port") {
            config.port = atoi(sargv[++i].c_str());
        } else {
            Usage();
            exit(-1);
        }
    }
}


int main(int argc, char** argv)
{
	/*   
    SampleProcess sampleProcess;
    Result ret = sampleProcess.InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("sample init resource failed");
        return FAILED;
    }
    */
    WebConfig config;
    ParseArgs(argc, argv, config);
    
    httplib::Server svr;
    SampleProcess sampleProcess;
    Result ret = sampleProcess.InitResource(config.modelPath, config.sentenceModelPath);
		
    svr.Post("/embedding", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string uuid = req.get_header_value("uuid");
        locker.lock();
        if (sessions.find(uuid) == sessions.end()) {
            sessions[uuid] = new ChatSession();
        }
        auto *session = sessions[uuid];
        locker.unlock();
        
        sampleProcess.InitStream();
        std::vector<float> data; 
        // std::string input = req.body;
        
        std::string err;
        Json reqData = Json::parse(req.body, err);
        // std::cout<<reqData.dump()<<std::endl;
        
        std::vector<std::string>inputs;
        for (auto &text:reqData["query"].array_items()){
        	inputs.push_back(text.string_value());
        	std::cout<<text.string_value()<<std::endl; 
        }     
		ret = sampleProcess.Process(inputs, data);
		if (ret != SUCCESS) {
		    ERROR_LOG("sample process failed");
		    return FAILED;
		}
		Json respJson = Json::object{
			{"code", 200},
			{"msg", ""},
			{"data", data}
        };
        session->output = respJson.dump();
        res.set_content(session->output, "application/json");
    });
    
    
    svr.Post("/rerank", [&](const httplib::Request &req, httplib::Response &res) {
        const std::string uuid = req.get_header_value("uuid");
        locker.lock();
        if (sessions.find(uuid) == sessions.end()) {
            sessions[uuid] = new ChatSession();
        }
        auto *session = sessions[uuid];
        locker.unlock();
        
        sampleProcess.InitStream();
        std::vector<float> data; 
        // std::string input = req.body;
        
        std::string err;
        Json reqData = Json::parse(req.body, err);
        // std::cout<<reqData.dump()<<std::endl;
        
        std::string query = reqData["query"].string_value();
        std::cout<<query<<std::endl; 
        
        std::vector<std::string>answers;
        for (auto &text:reqData["passages"].array_items()){
        	answers.push_back(text.string_value());
        	std::cout<<text.string_value()<<std::endl; 
        }
               
		ret = sampleProcess.Process(query, answers, data);
		if (ret != SUCCESS) {
		    ERROR_LOG("sample process failed!");
		    return FAILED;
		}
		Json respJson = Json::object{
			{"code", 200},
			{"msg", ""},
			{"data", data}
        };
        session->output = respJson.dump();
        res.set_content(session->output, "application/json");
    });
    
    svr.Get("/", [&](const httplib::Request &req, httplib::Response &res) {
    	res.set_content("hello, world", "text/plain");
    });
    
    // svr.set_mount_point("/", config.webPath);
    std::cout << ">>> please open http://127.0.0.1:" + std::to_string(config.port) + "\n";
    svr.listen("0.0.0.0", config.port);

    INFO_LOG("execute sample success");
    return SUCCESS;
}

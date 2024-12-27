#coding=utf-8
import numpy as np
import torch
from transformers import AutoTokenizer

from ais_bench.infer.interface import InferSession
from ais_bench.infer.interface import MemorySummary
from ais_bench.infer.summary import summary


# from torchvision.io import read_image
# import sentence_transformers
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import batch_to_device

MODEL_PATH = '/home/HwHiAiUser/Public/Models/bce-reranker-base_v1'

class PytorchInferencer:
    def __init__(self, model_path=''):
        self.model_path = MODEL_PATH  
        model = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = model

    def preprocess(self, texts):
        """预处理"""
        features = self.model(texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=512)
        print(features)
        features = [np.array(val) for key, val in features.items()]
        # features = [val[:512].reshape((1, 512)) for val in features]
        # features = np.array(features)
        # features = batch_to_device(features, target_device='cpu')
        # features = [np.random.randint(low=0, high=2, size=(1, 512)) for _ in range(2)]
        for i, feature in enumerate(features):
            print(feature.shape)
            feature.tofile(f"{i}.bin")
        return features

    def model_inference(self, model_input):
        """执行推理"""
        model_output = self.model(model_input)
        return model_output

    def postprocess(self, model_output):
        """后处理"""
        return model_output

    def e2e_inference(self, image_path):
        """端到端推理"""
        model_input = self.preprocess(image_path)
        model_output = self.model_inference(model_input)
        prediction = self.postprocess(model_output)
        return prediction


class OmInferencer(PytorchInferencer):
    def __init__(self, om_path, device_id=0):
        super(OmInferencer, self).__init__()
        self.session = InferSession(device_id=device_id, model_path=om_path)
        self.session.set_dynamic_shape("input_ids:1,10;attention_mask:1,10")
        # self.session.set_custom_outsize([10240000])
    
    def preprocess(self, texts):
        return super().preprocess(texts)

    def postprocess(self, model_output):
        return super().postprocess(torch.from_numpy(model_output))
    
    def model_inference(self, model_input):
        model_output = self.session.infer(feeds=model_input, mode='dymshape')[0]
        return model_output
    
    def e2e_inference(self, image_path):
        model_input = self.preprocess(image_path)
        model_output = self.model_inference(model_input)
        prediction = self.postprocess(model_output)
        print(prediction.size())
        return prediction
    
    def display_performance(self, batchsize=1, output_prefix=None, display_all_summary=True, multi_threads=False):
        s = self.session.summary()
        summary.npu_compute_time_list = [end_time - start_time for start_time, end_time in s.exec_time_list]
        summary.h2d_latency_list = MemorySummary.get_h2d_time_list()  # host to device
        summary.d2h_latency_list = MemorySummary.get_d2h_time_list()  # device to host
        summary.report(batchsize, output_prefix, display_all_summary, multi_threads)
 
 

class RerankInferencer(PytorchInferencer):
    def __init__(self, om_path, device_id=0):
        super(RerankInferencer, self).__init__()
        self.session = InferSession(device_id=device_id, model_path=om_path)
        # self.session.set_custom_outsize([10240000])
    
    def preprocess(self, texts):
        return super().preprocess(texts)

    def postprocess(self, model_output):
        return super().postprocess(torch.from_numpy(model_output))
    
    def model_inference(self, model_input):
        model_output = self.session.infer(feeds=model_input, mode='dymshape')[0]
        return model_output
    
    def e2e_inference(self, query="什么是熊猫", answers = ['猫和老虎是近亲关系', '大熊猫, 有时候叫做熊猫熊和小熊猫, 是一个中国特殊的熊猫种属', '熊猫之家']):
        pairs = [[ans, query] for ans in answers]
        model_input = self.preprocess(pairs)
        self.session.set_dynamic_shape("input_ids:3,10;attention_mask:3,10")
        model_output = self.model_inference(model_input)
        prediction = self.postprocess(model_output)
        print(prediction.size())
        return prediction
    
    def display_performance(self, batchsize=1, output_prefix=None, display_all_summary=True, multi_threads=False):
        s = self.session.summary()
        summary.npu_compute_time_list = [end_time - start_time for start_time, end_time in s.exec_time_list]
        summary.h2d_latency_list = MemorySummary.get_h2d_time_list()  # host to device
        summary.d2h_latency_list = MemorySummary.get_d2h_time_list()  # device to host
        summary.report(batchsize, output_prefix, display_all_summary, multi_threads)

inferencer = RerankInferencer('/home/HwHiAiUser/Public/BCE_APP/model/bce-rerank/bce_rerank_bs1-32_1-512_linux_aarch64.om')
print(inferencer.e2e_inference())

print("*"*50)

# filepath = "/home/HwHiAiUser/Public/Models/BCE_APP/model/output/2024_08_21-16_58_11/0_0.bin"
# filepath = "/home/HwHiAiUser/Public/Models/BCE_APP/data/1.bin"
# filepath = "1.bin"
# data = np.fromfile(filepath, dtype=np.float32)
# print(data)


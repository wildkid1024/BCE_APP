#include "tokenizer.h"

#include <sentencepiece_processor.h>


/*
int main2(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " tokenizer.txt" << std::endl;
        return 0;
    }
    std::string tokenizer_path = argv[1];
    std::unique_ptr<Tokenizer> tokenizer_(Tokenizer::createTokenizer(tokenizer_path));
    const std::string system_str = "You are a helpful assistant.";
    const std::string user_str = "<|endoftext|>";
    // const std::string query = "\n<|im_start|>system\n" + system_str + "<|im_end|>\n<|im_start|>\n" + user_str + "<|im_end|>\n<|im_start|>assistant\n";
    const std::string query = system_str + "\n" + user_str;
    auto tokens = tokenizer_->encode(query);

    std::string decode_str;
    printf("encode tokens = [ ");
    for (auto token : tokens) {
        decode_str += tokenizer_->decode(token);
    }
    printf("]\n");
    printf("decode str = %s\n", decode_str.c_str());
    return 0;
}
*/


int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " tokenizer.bpe.model" << std::endl;
        return 0;
    }
    std::string tokenizer_path = argv[1];
    sentencepiece::SentencePieceProcessor processor;
    const auto status = processor.Load(tokenizer_path);
	if (!status.ok()) {
	   std::cerr << status.ToString() << std::endl;
	   // error
	}
	
    const std::string system_str = "You are a helpful assistant.";
    const std::string user_str = "<|endoftext|>";
    // const std::string query = "\n<|im_start|>system\n" + system_str + "<|im_end|>\n<|im_start|>\n" + user_str + "<|im_end|>\n<|im_start|>assistant\n";
    const std::string quer2 = system_str + "\n" + user_str;
    
    
    const std::string query = "hello, what's your name";
    
    // processor.
    
    std::vector<std::string> pieces;
	processor.Encode(query, &pieces);
	for (const std::string &token : pieces) {
	  std::cout << token << std::endl;
	}
	
	std::vector<int> ids;
	processor.Encode(query, &ids);
	for (const int id : ids) {
	  std::cout << id << std::endl;
	}
	
	
	std::vector<std::string> pieces2 = { "▁This", "▁is", "▁a", "▁", "te", "st", "." };   // sequence of pieces
	std::string text;
	processor.Decode(pieces2, &text);
	std::cout << text << std::endl;

	std::vector<int> ids2 = { 0, 1, 2, 3, 250001};   // sequence of ids
	processor.Decode(ids2, &text);
	std::cout << text << std::endl;

    return 0;
}

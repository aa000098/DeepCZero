#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

class LlamaTokenizer {
private:
	// Vocabulary: token string -> id
	std::unordered_map<std::string, int> vocab;
	// Reverse vocabulary: id -> token string
	std::vector<std::string> id_to_token;
	// BPE merge rules: (pair) -> rank (lower rank = higher priority)
	std::unordered_map<std::string, int> merge_ranks;
	// Ordered merge pairs for BPE
	std::vector<std::pair<std::string, std::string>> merges;

	// Special token IDs
	int bos_token_id = -1;
	int eos_token_id = -1;
	int eot_token_id = -1;  // <|eot_id|>

	// GPT-2 byte-to-unicode mapping: raw byte -> unicode string (UTF-8)
	std::unordered_map<unsigned char, std::string> byte_to_token;
	// Reverse: unicode string -> raw byte
	std::unordered_map<std::string, unsigned char> token_to_byte;

	void init_byte_to_token();
	std::string merge_key(const std::string& a, const std::string& b) const;

	// BPE algorithm: merge tokens according to merge rules
	std::vector<std::string> bpe(const std::vector<std::string>& tokens) const;

public:
	LlamaTokenizer() = default;

	// Load from HuggingFace tokenizer.json
	void load(const std::string& tokenizer_json_path);

	// Encode text to token IDs
	std::vector<int> encode(const std::string& text) const;

	// Decode token IDs to text
	std::string decode(const std::vector<int>& ids) const;

	// Apply chat template for Llama 3.2 Instruct
	// Returns token IDs for: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
	std::vector<int> apply_chat_template(const std::string& user_message) const;

	// Decode a single token
	std::string decode_token(int id) const;

	// Getters
	int get_bos_token_id() const { return bos_token_id; }
	int get_eos_token_id() const { return eos_token_id; }
	int get_eot_token_id() const { return eot_token_id; }
	size_t vocab_size() const { return id_to_token.size(); }
};

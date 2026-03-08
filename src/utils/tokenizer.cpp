#include "utils/tokenizer.hpp"
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <regex>
#include <stdexcept>
#include <limits>

using json = nlohmann::json;

// Helper: encode a Unicode code point as a UTF-8 string
static std::string codepoint_to_utf8(uint32_t cp) {
	std::string result;
	if (cp < 0x80) {
		result += static_cast<char>(cp);
	} else if (cp < 0x800) {
		result += static_cast<char>(0xC0 | (cp >> 6));
		result += static_cast<char>(0x80 | (cp & 0x3F));
	} else if (cp < 0x10000) {
		result += static_cast<char>(0xE0 | (cp >> 12));
		result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
		result += static_cast<char>(0x80 | (cp & 0x3F));
	}
	return result;
}

void LlamaTokenizer::init_byte_to_token() {
	// GPT-2 bytes_to_unicode mapping:
	// Printable ASCII (33-126) and Latin-1 supplement (161-172, 174-255)
	// map to themselves. All other bytes (0-32, 127-160, 173) map to
	// chr(256+n) where n counts sequentially.
	std::vector<bool> is_direct(256, false);
	for (int i = 33; i <= 126; ++i) is_direct[i] = true;   // ASCII printable
	for (int i = 161; i <= 172; ++i) is_direct[i] = true;  // Latin-1 supplement
	for (int i = 174; i <= 255; ++i) is_direct[i] = true;  // Latin-1 supplement

	int offset = 0;
	for (int i = 0; i < 256; ++i) {
		unsigned char byte = static_cast<unsigned char>(i);
		std::string unicode_str;
		if (is_direct[i]) {
			unicode_str = codepoint_to_utf8(static_cast<uint32_t>(i));
		} else {
			unicode_str = codepoint_to_utf8(256 + offset);
			++offset;
		}
		byte_to_token[byte] = unicode_str;
		token_to_byte[unicode_str] = byte;
	}
}

std::string LlamaTokenizer::merge_key(const std::string& a, const std::string& b) const {
	return a + " " + b;
}

std::vector<std::string> LlamaTokenizer::bpe(const std::vector<std::string>& tokens) const {
	if (tokens.size() <= 1) return tokens;

	std::vector<std::string> word(tokens);

	while (word.size() > 1) {
		// Find the pair with lowest merge rank
		int best_rank = std::numeric_limits<int>::max();
		size_t best_pos = 0;
		bool found = false;

		for (size_t i = 0; i < word.size() - 1; ++i) {
			std::string key = merge_key(word[i], word[i + 1]);
			auto it = merge_ranks.find(key);
			if (it != merge_ranks.end() && it->second < best_rank) {
				best_rank = it->second;
				best_pos = i;
				found = true;
			}
		}

		if (!found) break;

		// Merge the best pair
		std::vector<std::string> new_word;
		for (size_t i = 0; i < word.size(); ++i) {
			if (i == best_pos) {
				new_word.push_back(word[i] + word[i + 1]);
				++i;  // Skip next token
			} else {
				new_word.push_back(word[i]);
			}
		}
		word = std::move(new_word);
	}

	return word;
}

void LlamaTokenizer::load(const std::string& tokenizer_json_path) {
	std::ifstream f(tokenizer_json_path);
	if (!f.is_open()) {
		throw std::runtime_error("Cannot open tokenizer file: " + tokenizer_json_path);
	}

	std::cout << "Loading tokenizer from: " << tokenizer_json_path << std::endl;
	json data = json::parse(f);

	// Load vocabulary from model.vocab
	const auto& model = data["model"];
	const auto& vocab_data = model["vocab"];

	// Find max id to size the id_to_token vector
	int max_id = 0;
	for (auto& [token, id] : vocab_data.items()) {
		int token_id = id.get<int>();
		if (token_id > max_id) max_id = token_id;
	}

	id_to_token.resize(max_id + 1);

	for (auto& [token, id] : vocab_data.items()) {
		int token_id = id.get<int>();
		vocab[token] = token_id;
		id_to_token[token_id] = token;
	}

	// Also load added_tokens (special tokens like <|begin_of_text|>)
	if (data.contains("added_tokens")) {
		for (const auto& added : data["added_tokens"]) {
			std::string content = added["content"].get<std::string>();
			int id = added["id"].get<int>();

			// Ensure id_to_token is large enough
			if (id >= static_cast<int>(id_to_token.size())) {
				id_to_token.resize(id + 1);
			}

			vocab[content] = id;
			id_to_token[id] = content;

			// Identify special tokens
			if (content == "<|begin_of_text|>") bos_token_id = id;
			else if (content == "<|end_of_text|>") eos_token_id = id;
			else if (content == "<|eot_id|>") eot_token_id = id;
		}
	}

	// Load merge rules
	if (model.contains("merges")) {
		const auto& merges_data = model["merges"];
		for (size_t i = 0; i < merges_data.size(); ++i) {
			std::string a, b;
			if (merges_data[i].is_array()) {
				// Array format: ["token1", "token2"]
				a = merges_data[i][0].get<std::string>();
				b = merges_data[i][1].get<std::string>();
			} else {
				// String format: "token1 token2"
				std::string merge_str = merges_data[i].get<std::string>();
				size_t space = merge_str.find(' ');
				if (space == std::string::npos) continue;
				a = merge_str.substr(0, space);
				b = merge_str.substr(space + 1);
			}
			merges.push_back({a, b});
			merge_ranks[merge_key(a, b)] = static_cast<int>(i);
		}
	}

	init_byte_to_token();

	std::cout << "Tokenizer loaded: vocab_size=" << id_to_token.size()
			  << ", merges=" << merges.size()
			  << ", bos=" << bos_token_id
			  << ", eos=" << eos_token_id
			  << ", eot=" << eot_token_id
			  << std::endl;
}

std::vector<int> LlamaTokenizer::encode(const std::string& text) const {
	if (text.empty()) return {};

	std::vector<int> result;

	// Llama 3 tokenizer uses a regex-based pre-tokenization pattern
	// The pattern splits on whitespace boundaries, numbers, special chars, etc.
	// Simplified approach: split text into chunks, then BPE each chunk

	// Pre-tokenization: split text into words/subwords
	// Llama 3 pattern (simplified): split before/after spaces, keeping spaces attached
	std::vector<std::string> pretokens;
	std::string current;

	for (size_t i = 0; i < text.size(); ++i) {
		char c = text[i];

		if (c == ' ' && !current.empty()) {
			pretokens.push_back(current);
			current.clear();
		}
		current += c;
	}
	if (!current.empty()) {
		pretokens.push_back(current);
	}

	// BPE encode each pretoken
	for (const auto& pretoken : pretokens) {
		// Convert raw bytes to GPT-2 unicode tokens
		std::string unicode_pretoken;
		std::vector<std::string> byte_tokens;
		for (unsigned char byte : pretoken) {
			auto map_it = byte_to_token.find(byte);
			const std::string& unicode_byte = (map_it != byte_to_token.end())
				? map_it->second : std::string(1, static_cast<char>(byte));
			unicode_pretoken += unicode_byte;
			byte_tokens.push_back(unicode_byte);
		}

		// Check if the whole unicode pretoken is a known token
		auto whole_it = vocab.find(unicode_pretoken);
		if (whole_it != vocab.end()) {
			result.push_back(whole_it->second);
			continue;
		}

		// Apply BPE merges
		auto merged = bpe(byte_tokens);

		// Convert merged tokens to IDs
		for (const auto& token : merged) {
			auto it = vocab.find(token);
			if (it != vocab.end()) {
				result.push_back(it->second);
			}
		}
	}

	return result;
}

std::string LlamaTokenizer::decode(const std::vector<int>& ids) const {
	std::string result;
	for (int id : ids) {
		result += decode_token(id);
	}
	return result;
}

std::string LlamaTokenizer::decode_token(int id) const {
	if (id < 0 || id >= static_cast<int>(id_to_token.size())) {
		return "";
	}
	const std::string& token = id_to_token[id];

	// Skip special tokens in output
	if (!token.empty() && token[0] == '<' && token.back() == '>') {
		return "";
	}

	// Convert GPT-2 unicode chars back to raw bytes
	std::string result;
	size_t i = 0;
	while (i < token.size()) {
		// Determine UTF-8 character length
		unsigned char c = static_cast<unsigned char>(token[i]);
		size_t char_len = 1;
		if ((c & 0xE0) == 0xC0) char_len = 2;
		else if ((c & 0xF0) == 0xE0) char_len = 3;
		else if ((c & 0xF8) == 0xF0) char_len = 4;

		if (i + char_len > token.size()) break;

		std::string utf8_char = token.substr(i, char_len);
		auto map_it = token_to_byte.find(utf8_char);
		if (map_it != token_to_byte.end()) {
			result += static_cast<char>(map_it->second);
		} else {
			result += utf8_char;
		}
		i += char_len;
	}
	return result;
}

std::vector<int> LlamaTokenizer::apply_chat_template(const std::string& user_message) const {
	// Llama 3.2 Instruct chat format:
	// <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
	std::vector<int> tokens;

	// Helper to add a special token by name
	auto add_special = [&](const std::string& name) {
		auto it = vocab.find(name);
		if (it != vocab.end()) {
			tokens.push_back(it->second);
		}
	};

	// <|begin_of_text|>
	add_special("<|begin_of_text|>");

	// <|start_header_id|>
	add_special("<|start_header_id|>");

	// "user"
	auto user_tokens = encode("user");
	tokens.insert(tokens.end(), user_tokens.begin(), user_tokens.end());

	// <|end_header_id|>
	add_special("<|end_header_id|>");

	// "\n\n"
	auto newlines = encode("\n\n");
	tokens.insert(tokens.end(), newlines.begin(), newlines.end());

	// User message
	auto msg_tokens = encode(user_message);
	tokens.insert(tokens.end(), msg_tokens.begin(), msg_tokens.end());

	// <|eot_id|>
	add_special("<|eot_id|>");

	// <|start_header_id|>
	add_special("<|start_header_id|>");

	// "assistant"
	auto asst_tokens = encode("assistant");
	tokens.insert(tokens.end(), asst_tokens.begin(), asst_tokens.end());

	// <|end_header_id|>
	add_special("<|end_header_id|>");

	// "\n\n"
	tokens.insert(tokens.end(), newlines.begin(), newlines.end());

	return tokens;
}

#include "deepczero.hpp"
#include "utils/generate.hpp"
#include "utils/tokenizer.hpp"

#include <iostream>
#include <string>
#include <cstdlib>

int main() {
	dcz::UsingConfig eval_mode("train", false);
	dcz::UsingConfig no_grad("enable_backprop", false);

	std::string home = std::getenv("HOME");

	// 1. Load tokenizer
	std::string tokenizer_path = home + "/.deepczero/weights/tokenizer.json";
	LlamaTokenizer tokenizer;
	tokenizer.load(tokenizer_path);
	std::cout << "Tokenizer loaded. Vocab size: " << tokenizer.vocab_size() << std::endl;

	// 2. Create model
	std::cout << "\nCreating Llama 3.2 1B model..." << std::endl;
	LlamaForCausalLM model(
		128256,  // vocab_size
		2048,    // hidden_size
		16,      // num_layers
		32,      // num_heads
		8,       // num_kv_heads
		8192,    // intermediate_size
		512,     // max_position_embeddings
		500000.0f,  // rope_theta
		1e-5f    // rms_norm_eps
	);

	// 3. Load weights
	std::string weights_path = home + "/.deepczero/weights/llama-3.2-1b-instruct.npz";
	model.load_weights(weights_path);

#ifdef USE_SYCL
	// Move model to GPU
	dcz::SYCLContext::get().print_device_info();
	std::cout << "\nMoving model to GPU..." << std::endl;
	model.to(dcz::sycl());
	std::cout << "Model moved to GPU." << std::endl;
#endif

	// 4. Tokenize prompt using chat template
	std::string prompt = "What is the capital of France?";
	std::cout << "\nPrompt: " << prompt << std::endl;

	std::vector<int> token_ids = tokenizer.apply_chat_template(prompt);
	std::cout << "Token IDs (" << token_ids.size() << " tokens): ";
	for (int id : token_ids) std::cout << id << " ";
	std::cout << std::endl;

	// 5. Generate
	GenerationConfig config;
	config.max_new_tokens = 64;
	config.do_sample = false;  // greedy
	config.eos_token_id = tokenizer.get_eot_token_id();

	{
		dcz::UsingConfig profile_mode("profile", true);
		std::cout << "\nGenerating..." << std::endl;
		auto generated = generate(model, token_ids, config);

		// 6. Decode and print
		std::cout << "\nGenerated tokens (" << generated.size() << "): ";
		for (int id : generated) std::cout << id << " ";
		std::cout << std::endl;

		std::string output = tokenizer.decode(generated);
		std::cout << "\nOutput: " << output << std::endl;
	}
	return 0;
}

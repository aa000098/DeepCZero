#include "function/normalization_functions.hpp"
#include "container/tensor/tensor_all.hpp"

#include <cmath>

using namespace tensor;

Variable function::BatchNorm2dFunc::forward(const std::vector<Variable>& xs) {
	dcz::Device orig_device = xs[0].device();
	const Tensor<> x = !orig_device.is_cpu() ? xs[0].data().cpu() : xs[0].data();
	const Tensor<> gamma = !orig_device.is_cpu() ? xs[1].data().cpu() : xs[1].data();
	const Tensor<> beta = !orig_device.is_cpu() ? xs[2].data().cpu() : xs[2].data();

	auto shape = x.get_shape();
	N = shape[0]; C = shape[1]; H = shape[2]; W = shape[3];
	float M = static_cast<float>(N * H * W);

	Tensor<> mu, var;
	if (is_training) {
		// Compute batch mean per channel
		mu = x.sum({0, 2, 3}) / M;  // [C]

		// Compute batch variance per channel
		Tensor<> mu_4d({1, C, 1, 1}, mu.raw_data());
		Tensor<> mu_bc = broadcast_to(mu_4d, {N, C, H, W}).contiguous();
		Tensor<> diff = x - mu_bc;
		Tensor<> diff_sq = diff * diff;
		var = diff_sq.sum({0, 2, 3}) / M;  // [C]
	} else {
		mu = !orig_device.is_cpu() ? xs[3].data().cpu() : xs[3].data();
		var = !orig_device.is_cpu() ? xs[4].data().cpu() : xs[4].data();
	}

	// Compute inv_std per channel
	std::vector<float> inv_std_data(C);
	for (size_t c = 0; c < C; ++c)
		inv_std_data[c] = 1.0f / std::sqrt(var.raw_data()[c] + eps);

	saved_mean = Tensor<>({C}, mu.raw_data());
	saved_inv_std = Tensor<>({C}, inv_std_data);

	// Broadcast for element-wise ops (contiguous for MKL compatibility)
	Tensor<> mu_4d({1, C, 1, 1}, mu.raw_data());
	Tensor<> inv_std_4d({1, C, 1, 1}, inv_std_data);
	Tensor<> gamma_4d({1, C, 1, 1}, gamma.raw_data());
	Tensor<> beta_4d({1, C, 1, 1}, beta.raw_data());

	Tensor<> mu_bc = broadcast_to(mu_4d, {N, C, H, W}).contiguous();
	Tensor<> inv_std_bc = broadcast_to(inv_std_4d, {N, C, H, W}).contiguous();
	Tensor<> gamma_bc = broadcast_to(gamma_4d, {N, C, H, W}).contiguous();
	Tensor<> beta_bc = broadcast_to(beta_4d, {N, C, H, W}).contiguous();

	// x_hat = (x - mu) * inv_std
	Tensor<> x_hat = (x - mu_bc) * inv_std_bc;

	// y = gamma * x_hat + beta
	Tensor<> y = gamma_bc * x_hat + beta_bc;

	if (!orig_device.is_cpu()) y = y.to(orig_device);
	return Variable(y);
}

std::vector<Variable> function::BatchNorm2dFunc::backward(const Variable& gy) {
	dcz::Device orig_device = gy.device();
	const Tensor<> x = !orig_device.is_cpu() ? inputs[0]->data.cpu() : inputs[0]->data;
	const Tensor<> gamma = !orig_device.is_cpu() ? inputs[1]->data.cpu() : inputs[1]->data;
	const Tensor<> gy_data = !orig_device.is_cpu() ? gy.data().cpu() : gy.data();

	float M = static_cast<float>(N * H * W);

	// Reconstruct x_hat from saved statistics (contiguous for MKL compatibility)
	Tensor<> mu_4d({1, C, 1, 1}, saved_mean.raw_data());
	Tensor<> inv_std_4d({1, C, 1, 1}, saved_inv_std.raw_data());
	Tensor<> mu_bc = broadcast_to(mu_4d, {N, C, H, W}).contiguous();
	Tensor<> inv_std_bc = broadcast_to(inv_std_4d, {N, C, H, W}).contiguous();
	Tensor<> x_hat = (x - mu_bc) * inv_std_bc;

	// dgamma = sum(gy * x_hat, axes={0,2,3})
	Tensor<> dgamma = (gy_data * x_hat).sum({0, 2, 3});  // [C]

	// dbeta = sum(gy, axes={0,2,3})
	Tensor<> dbeta = gy_data.sum({0, 2, 3});  // [C]

	// Efficient dx computation
	Tensor<> gamma_4d({1, C, 1, 1}, gamma.raw_data());
	Tensor<> gamma_bc = broadcast_to(gamma_4d, {N, C, H, W}).contiguous();
	Tensor<> dx_hat = gy_data * gamma_bc;

	Tensor<> sum_dxhat = dx_hat.sum({0, 2, 3});              // [C]
	Tensor<> sum_dxhat_xhat = (dx_hat * x_hat).sum({0, 2, 3}); // [C]

	// Broadcast sums to [N,C,H,W] (contiguous for MKL compatibility)
	Tensor<> s1_4d({1, C, 1, 1}, sum_dxhat.raw_data());
	Tensor<> s2_4d({1, C, 1, 1}, sum_dxhat_xhat.raw_data());
	Tensor<> s1_bc = broadcast_to(s1_4d, {N, C, H, W}).contiguous();
	Tensor<> s2_bc = broadcast_to(s2_4d, {N, C, H, W}).contiguous();

	// dx = inv_std/M * (M*dx_hat - sum(dx_hat) - x_hat*sum(dx_hat*x_hat))
	Tensor<> dx = inv_std_bc / M * (dx_hat * M - s1_bc - x_hat * s2_bc);

	if (!orig_device.is_cpu()) {
		dx = dx.to(orig_device);
		dgamma = dgamma.to(orig_device);
		dbeta = dbeta.to(orig_device);
	}
	return { Variable(dx), Variable(dgamma), Variable(dbeta) };
}

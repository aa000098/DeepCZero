#include "function/loss_functions.hpp"
#include "container/tensor/tensor_all.hpp"


Variable function::MeanSquaredError::forward(const std::vector<Variable>& xs) {
	const Tensor<>& a = xs[0].data();
	const Tensor<>& b = xs[1].data();

	const Tensor<> diff = a - b; 
	float scale = static_cast<float>(diff.size());
	const Tensor<> result = (diff^2.0f).sum() / scale;
	return Variable(result);
}

std::vector<Variable> function::MeanSquaredError::backward(const Variable& gy) {
	const Variable& x0 = inputs[0];
	const Variable& x1 = inputs[1];
	Variable diff = x0 - x1;

	float scale = 2.0f / static_cast<float>(diff.size());
	Variable gx0 = gy * diff * scale;
	Variable gx1 = -gx0;
	
	return {gx0, gx1};
}


Variable function::SoftmaxCrossEntropyError::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	const Tensor<>& t = xs[1].data();

#ifdef USE_SYCL
	if (x.is_device()) {
		Tensor<> result = tensor::softmax_cross_entropy_forward_sycl(x, t);
		return Variable(result);
	}
#endif

	float N = x.get_shape()[0];

	// logsumexp
	Tensor<> m = x.max({1}, true);
	Tensor<> y = exp(x - m);
	Tensor<> sum_y = y.sum({1}, true);
	sum_y = log(sum_y);

	Tensor<> log_z = m + sum_y;
	Tensor<> log_p = x - log_z;

	float loss = 0.0f;
	for (size_t i = 0; i < N; i++) {
		size_t label = static_cast<size_t>(t({i}));
		loss += -log_p({i, label});
	}

	loss /= N;

	return Variable(Tensor<>(1, loss));
}

std::vector<Variable> function::SoftmaxCrossEntropyError::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	const Variable& t = inputs[1];

#ifdef USE_SYCL
	if (x.data().is_device()) {
		// Get scalar gy value
		float gy_val = gy.data().cpu().raw_data()[0];
		Tensor<> dx = tensor::softmax_cross_entropy_backward_sycl(
			x.data(), t.data(), gy_val);
		return { Variable(dx) };
	}
#endif

	size_t N = x.shape()[0];
	size_t C = x.shape()[1];

	Variable y = softmax(x);

	Tensor<> t_onehot({N, C}, 0.0f);

	for (size_t i = 0; i < N; i++) {
		size_t label = t.data()({i});
		t_onehot({i, label}) = 1.0f;
	}

	Variable dx = (y - Variable(t_onehot)) * (gy / static_cast<float>(N));

	return { dx };
}


// BinaryCrossEntropy: numerically stable BCE with logits
// loss = mean( max(x,0) - x*t + log(1 + exp(-|x|)) )
Variable function::BinaryCrossEntropy::forward(const std::vector<Variable>& xs) {
	const Tensor<>& x = xs[0].data();
	const Tensor<>& t = xs[1].data();
	size_t N = x.size();

	Tensor<> abs_x = tensor::abs(x);
	Tensor<> relu_x = tensor::maximum(x, 0.0f);
	Tensor<> log_term = tensor::log(tensor::exp(-abs_x) + 1.0f);

	Tensor<> loss_elem;
	if (pos_weight == 1.0f) {
		loss_elem = relu_x - x * t + log_term;
	} else {
		Tensor<> weight = t * (pos_weight - 1.0f) + 1.0f;
		loss_elem = relu_x - x * t * pos_weight + weight * log_term;
	}

	Tensor<> result = loss_elem.sum() / static_cast<float>(N);
	return Variable(result);
}

std::vector<Variable> function::BinaryCrossEntropy::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	const Variable& t = inputs[1];
	size_t N = x.size();

	Variable sig_x = sigmoid(x);
	Variable dx;
	if (pos_weight == 1.0f) {
		dx = (sig_x - t) * (gy / static_cast<float>(N));
	} else {
		Variable weight = t * (pos_weight - 1.0f) + 1.0f;
		dx = (sig_x * weight - t * pos_weight) * (gy / static_cast<float>(N));
	}
	return { dx };
}


// Abs
Variable function::Abs::forward(const std::vector<Variable>& xs) {
	return Variable(tensor::abs(xs[0].data()));
}

std::vector<Variable> function::Abs::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	Variable sign_var(tensor::sign(x.data()));
	return { gy * sign_var };
}


// Clamp
Variable function::Clamp::forward(const std::vector<Variable>& xs) {
	return Variable(tensor::clamp(xs[0].data(), min_val, max_val));
}

std::vector<Variable> function::Clamp::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	dcz::Device orig_device = x.device();
	Tensor<> x_cpu = orig_device.is_cpu() ? x.data() : x.data().cpu();

	const auto& x_data = x_cpu.raw_data();
	size_t N = x.size();

	std::vector<float> mask_data(N);
	for (size_t i = 0; i < N; i++)
		mask_data[i] = (x_data[i] > min_val && x_data[i] < max_val) ? 1.0f : 0.0f;

	Tensor<> mask_tensor(x.shape(), mask_data);
	if (!orig_device.is_cpu()) mask_tensor = mask_tensor.to(orig_device);
	Variable mask(mask_tensor);
	return { gy * mask };
}


// ============================================================
// CIoU: Complete IoU loss for bounding box regression
// inputs: pred [N, 4], target [N, 4] in (cx, cy, w, h) format
// output: scalar loss = mean(1 - CIoU)
// ============================================================
Variable function::CIoU::forward(const std::vector<Variable>& xs) {
	dcz::Device orig_device = xs[0].device();
	const Tensor<>& pred = orig_device.is_cpu() ? xs[0].data() : xs[0].data().cpu();
	const Tensor<>& target = orig_device.is_cpu() ? xs[1].data() : xs[1].data().cpu();
	size_t N = pred.get_shape()[0];

	if (N == 0) {
		Variable result(Tensor<>(1, 0.0f));
		return orig_device.is_cpu() ? result : result.to(orig_device);
	}

	const auto& p = pred.raw_data();
	const auto& t = target.raw_data();

	float pi = 3.14159265358979f;
	float four_over_pi_sq = 4.0f / (pi * pi);
	float total_loss = 0.0f;

	for (size_t i = 0; i < N; i++) {
		float pcx = p[i*4+0], pcy = p[i*4+1], pw = p[i*4+2], ph = p[i*4+3];
		float tcx = t[i*4+0], tcy = t[i*4+1], tw = t[i*4+2], th = t[i*4+3];

		// Corner coords
		float px1 = pcx - pw*0.5f, py1 = pcy - ph*0.5f;
		float px2 = pcx + pw*0.5f, py2 = pcy + ph*0.5f;
		float tx1 = tcx - tw*0.5f, ty1 = tcy - th*0.5f;
		float tx2 = tcx + tw*0.5f, ty2 = tcy + th*0.5f;

		// Intersection
		float ix1 = std::max(px1, tx1), iy1 = std::max(py1, ty1);
		float ix2 = std::min(px2, tx2), iy2 = std::min(py2, ty2);
		float inter_w = std::max(0.0f, ix2 - ix1);
		float inter_h = std::max(0.0f, iy2 - iy1);
		float inter_area = inter_w * inter_h;

		float pred_area = pw * ph;
		float tgt_area = tw * th;
		float union_area = pred_area + tgt_area - inter_area + 1e-7f;
		float iou = inter_area / union_area;

		// Enclosing box diagonal^2
		float ex1 = std::min(px1, tx1), ey1 = std::min(py1, ty1);
		float ex2 = std::max(px2, tx2), ey2 = std::max(py2, ty2);
		float c_sq = (ex2 - ex1) * (ex2 - ex1) + (ey2 - ey1) * (ey2 - ey1) + 1e-7f;

		// Center distance^2
		float rho_sq = (pcx - tcx) * (pcx - tcx) + (pcy - tcy) * (pcy - tcy);

		// Aspect ratio
		float v = four_over_pi_sq *
			std::pow(std::atan(tw / (th + 1e-7f)) - std::atan(pw / (ph + 1e-7f)), 2.0f);
		float alpha = v / (1.0f - iou + v + 1e-7f);

		float ciou = iou - rho_sq / c_sq - alpha * v;
		total_loss += 1.0f - ciou;
	}

	Variable result(Tensor<>(1, total_loss / static_cast<float>(N)));
	return orig_device.is_cpu() ? result : result.to(orig_device);
}

std::vector<Variable> function::CIoU::backward(const Variable& gy) {
	const Variable& pred_var = inputs[0];
	const Variable& target_var = inputs[1];
	dcz::Device orig_device = pred_var.device();
	const Tensor<> pred = orig_device.is_cpu() ? pred_var.data() : pred_var.data().cpu();
	const Tensor<> target = orig_device.is_cpu() ? target_var.data() : target_var.data().cpu();
	size_t N = pred.get_shape()[0];

	if (N == 0) {
		Variable gp(Tensor<>(pred.get_shape(), 0.0f));
		Variable gt(Tensor<>(target.get_shape(), 0.0f));
		if (!orig_device.is_cpu()) { gp = gp.to(orig_device); gt = gt.to(orig_device); }
		return { gp, gt };
	}

	Tensor<> gy_cpu = orig_device.is_cpu() ? gy.data() : gy.data().cpu();
	float gy_val = gy_cpu.raw_data()[0];
	float scale = gy_val / static_cast<float>(N);

	const auto& p = pred.raw_data();
	const auto& t = target.raw_data();

	float pi = 3.14159265358979f;
	float four_over_pi_sq = 4.0f / (pi * pi);

	std::vector<float> grad_pred(N * 4, 0.0f);

	for (size_t i = 0; i < N; i++) {
		float pcx = p[i*4+0], pcy = p[i*4+1], pw = p[i*4+2], ph = p[i*4+3];
		float tcx = t[i*4+0], tcy = t[i*4+1], tw = t[i*4+2], th = t[i*4+3];

		// Corner coords
		float px1 = pcx - pw*0.5f, py1 = pcy - ph*0.5f;
		float px2 = pcx + pw*0.5f, py2 = pcy + ph*0.5f;
		float tx1 = tcx - tw*0.5f, ty1 = tcy - th*0.5f;
		float tx2 = tcx + tw*0.5f, ty2 = tcy + th*0.5f;

		// Intersection
		float ix1 = std::max(px1, tx1), iy1 = std::max(py1, ty1);
		float ix2 = std::min(px2, tx2), iy2 = std::min(py2, ty2);
		float inter_w = std::max(0.0f, ix2 - ix1);
		float inter_h = std::max(0.0f, iy2 - iy1);
		float inter_area = inter_w * inter_h;

		float pred_area = pw * ph;
		float tgt_area = tw * th;
		float union_area = pred_area + tgt_area - inter_area + 1e-7f;
		float iou = inter_area / union_area;

		// Enclosing box
		float ex1 = std::min(px1, tx1), ey1 = std::min(py1, ty1);
		float ex2 = std::max(px2, tx2), ey2 = std::max(py2, ty2);
		float c_sq = (ex2 - ex1) * (ex2 - ex1) + (ey2 - ey1) * (ey2 - ey1) + 1e-7f;

		float rho_sq = (pcx - tcx) * (pcx - tcx) + (pcy - tcy) * (pcy - tcy);

		// Aspect ratio
		float atan_t = std::atan(tw / (th + 1e-7f));
		float atan_p = std::atan(pw / (ph + 1e-7f));
		float v = four_over_pi_sq * (atan_t - atan_p) * (atan_t - atan_p);
		float alpha = v / (1.0f - iou + v + 1e-7f);

		// loss = 1 - iou + rho_sq/c_sq + alpha*v
		// d(loss)/d(pred) = -d(iou)/d(pred) + d(rho_sq/c_sq)/d(pred) + d(alpha*v)/d(pred)

		// --- IoU gradient ---
		// d(iou)/d(pred_cx):
		// iou = inter_area / union_area
		// d(iou)/dx = (d(inter)/dx * union - inter * d(union)/dx) / union^2
		// d(union)/dx = d(pred_area)/dx - d(inter)/dx

		// Intersection partial derivatives w.r.t. pred corners
		// inter_w = max(0, min(px2,tx2) - max(px1,tx1))
		// d(inter_w)/d(px1) = -(px1 >= tx1 ? 1 : 0) if inter_w > 0
		// d(inter_w)/d(px2) = (px2 <= tx2 ? 1 : 0) if inter_w > 0

		float dinter_w_dpx1 = (inter_w > 0 && px1 >= tx1) ? -1.0f : 0.0f;
		float dinter_w_dpx2 = (inter_w > 0 && px2 <= tx2) ? 1.0f : 0.0f;
		float dinter_h_dpy1 = (inter_h > 0 && py1 >= ty1) ? -1.0f : 0.0f;
		float dinter_h_dpy2 = (inter_h > 0 && py2 <= ty2) ? 1.0f : 0.0f;

		// d(inter_area)/d(pcx): px1=pcx-pw/2, px2=pcx+pw/2, both depend on pcx
		float dia_dpcx = (dinter_w_dpx1 + dinter_w_dpx2) * inter_h;
		float dia_dpcy = (dinter_h_dpy1 + dinter_h_dpy2) * inter_w;

		// d(inter_area)/d(pw): d(px1)/d(pw)=-0.5, d(px2)/d(pw)=0.5
		float d_inter_dpw = (dinter_w_dpx1 * (-0.5f) + dinter_w_dpx2 * 0.5f) * inter_h;
		float d_inter_dph = (dinter_h_dpy1 * (-0.5f) + dinter_h_dpy2 * 0.5f) * inter_w;

		// d(union)/d(pred) = d(pred_area)/d(pred) - d(inter)/d(pred)
		// d(pred_area)/d(pw) = ph, d(pred_area)/d(ph) = pw
		float d_union_dpcx = -dia_dpcx;
		float d_union_dpcy = -dia_dpcy;
		float d_union_dpw = ph - d_inter_dpw;
		float d_union_dph = pw - d_inter_dph;

		// d(iou)/d(pred) = (d_inter * union - inter * d_union) / union^2
		float u_sq = union_area * union_area;
		float diou_dpcx = (dia_dpcx * union_area - inter_area * d_union_dpcx) / u_sq;
		float diou_dpcy = (dia_dpcy * union_area - inter_area * d_union_dpcy) / u_sq;
		float diou_dpw = (d_inter_dpw * union_area - inter_area * d_union_dpw) / u_sq;
		float diou_dph = (d_inter_dph * union_area - inter_area * d_union_dph) / u_sq;

		// --- Distance penalty gradient ---
		// rho_sq/c_sq, d/d(pcx) = 2*(pcx-tcx)/c_sq - rho_sq * d(c_sq)/d(pcx) / c_sq^2
		// c_sq = (ex2-ex1)^2 + (ey2-ey1)^2
		// Enclosing box: ex1 = min(px1,tx1), ex2 = max(px2,tx2)
		float dex1_dpx1 = (px1 <= tx1) ? 1.0f : 0.0f;
		float dex2_dpx2 = (px2 >= tx2) ? 1.0f : 0.0f;
		float dey1_dpy1 = (py1 <= ty1) ? 1.0f : 0.0f;
		float dey2_dpy2 = (py2 >= ty2) ? 1.0f : 0.0f;

		float ew = ex2 - ex1, eh = ey2 - ey1;

		// d(c_sq)/d(pcx) = 2*ew*(dex2_dpx2 - dex1_dpx1) * d(px)/d(pcx)
		// d(px1)/d(pcx) = 1, d(px2)/d(pcx) = 1
		float dc_sq_dpcx = 2.0f * ew * (dex2_dpx2 * 1.0f - dex1_dpx1 * 1.0f);
		float dc_sq_dpcy = 2.0f * eh * (dey2_dpy2 * 1.0f - dey1_dpy1 * 1.0f);
		float dc_sq_dpw = 2.0f * ew * (dex2_dpx2 * 0.5f - dex1_dpx1 * (-0.5f));
		float dc_sq_dph = 2.0f * eh * (dey2_dpy2 * 0.5f - dey1_dpy1 * (-0.5f));

		float drho_dpcx = 2.0f * (pcx - tcx) / c_sq - rho_sq * dc_sq_dpcx / (c_sq * c_sq);
		float drho_dpcy = 2.0f * (pcy - tcy) / c_sq - rho_sq * dc_sq_dpcy / (c_sq * c_sq);
		float drho_dpw = -rho_sq * dc_sq_dpw / (c_sq * c_sq);
		float drho_dph = -rho_sq * dc_sq_dph / (c_sq * c_sq);

		// --- Aspect ratio gradient (only through atan(pw/ph)) ---
		// d(alpha*v)/d(pw) = alpha * d(v)/d(pw)
		// d(v)/d(pw) = 2 * four_over_pi_sq * (atan_t - atan_p) * (-d(atan_p)/d(pw))
		// d(atan(pw/ph))/d(pw) = 1/(1+(pw/ph)^2) * (1/ph)
		// d(atan(pw/ph))/d(ph) = 1/(1+(pw/ph)^2) * (-pw/ph^2)
		float r = pw / (ph + 1e-7f);
		float datan_dpw = 1.0f / (1.0f + r * r) / (ph + 1e-7f);
		float datan_dph = -r / (1.0f + r * r) / (ph + 1e-7f);

		float dv_dpw = four_over_pi_sq * 2.0f * (atan_t - atan_p) * (-datan_dpw);
		float dv_dph = four_over_pi_sq * 2.0f * (atan_t - atan_p) * (-datan_dph);

		float dav_dpw = alpha * dv_dpw;
		float dav_dph = alpha * dv_dph;

		// Total gradient: d(loss)/d(pred) = -d(iou) + d(rho) + d(av)
		// loss = 1 - iou + rho_sq/c_sq + alpha*v
		grad_pred[i*4+0] = (-diou_dpcx + drho_dpcx) * scale;
		grad_pred[i*4+1] = (-diou_dpcy + drho_dpcy) * scale;
		grad_pred[i*4+2] = (-diou_dpw + drho_dpw + dav_dpw) * scale;
		grad_pred[i*4+3] = (-diou_dph + drho_dph + dav_dph) * scale;
	}

	Variable grad_pred_var(Tensor<>(pred.get_shape(), grad_pred));
	if (!orig_device.is_cpu()) grad_pred_var = grad_pred_var.to(orig_device);
	return { grad_pred_var };
}


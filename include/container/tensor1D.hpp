
namespace tensor {
	template<typename T>
	class Tensor1D : public TensorBase<T> {
	private:
		std::vector<T> data;
	public:
		Tensor1D(size_t len, T init = T()) : data(len, init) {};
		size_t size() const override { return data.size(); };
		size_t ndim() const override { return 1; };
		std::vector<size_t> shape() const override { return {data.size()}; };
	};
}

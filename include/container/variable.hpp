#pragma once

#include <memory>

class Function; // forward declaration

class Variable {
protected:
	//TODO: extend to tensor
    float data;
    float grad;
    std::shared_ptr<Function> creator;
    bool requires_grad;

public:
    Variable(float data, bool requires_grad = true);

	float get_data() { return data; };
	float get_grad() { return grad; };

    void set_creator(std::shared_ptr<Function> func) { creator = func; };
	std::shared_ptr<Function> get_creator() { return creator; };
    void backward();
    void show() const;
};
	

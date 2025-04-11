#pragma once

#include <memory>

class Function; // forward declaration

class Variable {

    float data;
    float grad;
    std::shared_ptr<Function> creator;
    bool requires_grad;

public:
    Variable(float data, bool requires_grad = true);

    void set_creator(std::shared_ptr<Function> func);
    void backward();
    void show() const;
};
	

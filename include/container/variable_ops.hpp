#pragma once

#include "container/variable.hpp"
#include "ops/ops.hpp"

// Variable + Variable
inline Variable operator+(const Variable& lhs, const Variable& rhs) {
    return add(lhs, rhs);
}

// Variable + scalar
inline Variable operator+(const Variable& lhs, float rhs) {
    return add(lhs, rhs);
}

// scalar + Variable
inline Variable operator+(float lhs, const Variable& rhs) {
    return add(lhs, rhs);
}

// Variable - Variable
inline Variable operator-(const Variable& lhs, const Variable& rhs) {
    return sub(lhs, rhs);
}

// Variable - scalar
inline Variable operator-(const Variable& lhs, float rhs) {
    return sub(lhs, rhs);
}

// scalar - Variable
inline Variable operator-(float lhs, const Variable& rhs) {
    return sub(lhs, rhs);
}

// Unary minus
inline Variable operator-(const Variable& x) {
    return neg(x);
}

// Variable * Variable
inline Variable operator*(const Variable& lhs, const Variable& rhs) {
    return mul(lhs, rhs);
}

// Variable * scalar
inline Variable operator*(const Variable& lhs, float rhs) {
    return mul(lhs, rhs);
}

// scalar * Variable
inline Variable operator*(float lhs, const Variable& rhs) {
	return mul(lhs, rhs);
}

// Variable / Variable
inline Variable operator/(const Variable& lhs, const Variable& rhs) {
    return div(lhs, rhs);
}

// Variable / scalar
inline Variable operator/(const Variable& lhs, float rhs) {
    return div(lhs, rhs);
}

// scalar / Variable
inline Variable operator/(float lhs, const Variable& rhs) {
    return div(lhs, rhs);
}

// Variable ^ scalar (거듭제곱)
inline Variable operator^(const Variable& lhs, float rhs) {
    return pow(lhs, rhs);
}

// TODO: inplace operation need to be have same memory space  
inline Variable& operator+=(Variable& lhs, const Variable& rhs) {
    lhs = add(lhs, rhs);
    return lhs;
}

inline Variable& operator-=(Variable& lhs, const Variable& rhs) {
    lhs = sub(lhs, rhs);
    return lhs;
}

inline Variable& operator/=(Variable& lhs, const Variable& rhs) {
    lhs = div(lhs, rhs);
    return lhs;
}

inline Variable& operator+=(Variable& lhs, float rhs) {
    lhs = add(lhs, rhs);
    return lhs;
}

inline Variable& operator-=(Variable& lhs, float rhs) {
    lhs = sub(lhs, rhs);
    return lhs;
}

inline Variable& operator*=(Variable& lhs, float rhs) {
    lhs = mul(lhs, rhs);
    return lhs;
}

inline Variable& operator/=(Variable& lhs, float rhs) {
    lhs = div(lhs, rhs);
    return lhs;
}

inline Variable& operator^=(Variable& lhs, float rhs) {
    lhs = pow(lhs, rhs);
    return lhs;
}



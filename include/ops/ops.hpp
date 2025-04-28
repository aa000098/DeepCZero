#pragma once

class Variable;

Variable square(const Variable &x);
Variable exp(const Variable &x);
Variable add(const Variable &a, const Variable &b);
Variable add(const Variable &a, const float &b);
Variable add(const float &a, const Variable &b);
Variable mul(const Variable &a, const Variable &b);
Variable mul(const Variable &a, const float &b);
Variable mul(const float &a, const Variable &b);

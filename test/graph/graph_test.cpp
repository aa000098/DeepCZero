#include "graph/graph.hpp"
#include "ops/ops.hpp"
#include "container/variable.hpp"

#include <iostream>
#include <vector>
#include <memory>

 
int main() {
    Variable x({2});
	Variable a = square(x);
	Variable b = exp(a);
	Variable c1 = square(b);
	Variable c2 = add(b, x);
    Variable d = add(c1, c2);
	Variable y = square(d);

    std::cout << "[Forward done]" << std::endl;

	y.backward();
	std::cout << "\n[Result y]: (9.22675e+06)\n";
	y.show();

	std::cout << "\n[Input x auto-backward result]: (1.46873e+08)\n";
	x.show();

    // 그래프 생성
    Graph graph;
    graph.build_from(y.get_creator().get());

    // 그래프 출력
    graph.print_graph();

    // Topo 순서 출력 (optional)
    std::cout << "\n=== Topological Order ===\n";
    auto topo = graph.get_topo_order();
    for (auto* f : topo) {
        std::cout << f << " | " << typeid(*f).name() << std::endl;
    }
}

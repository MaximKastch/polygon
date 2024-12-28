#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <limits>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <stack>

#include "graph.cpp"

Graph* load_graph_from_file(std::string file_name) {
    std::string line;
    std::ifstream in(file_name);
    Graph* graph = new Graph();
    while(std::getline(in, line))
    {
        std::vector<std::string> graph_node_parts = splitString(line, ':');
        Node* current_node = graph->getOrAddNodeByCoordinatesLine(graph_node_parts[0]);
        std::vector<std::string> neighbours_strings = splitString(graph_node_parts[1], ';');
        for (std::string neighbour_string : neighbours_strings) {
            auto neighbour_parts = splitString(neighbour_string, ',');
            auto neighbour_key = neighbour_parts[0] + "," + neighbour_parts[1];
            auto neighbour_weight = stod(neighbour_parts[2]);
            Node* child = graph->getOrAddNodeByCoordinatesLine(neighbour_key);
            current_node->nodes.emplace_back(child, neighbour_weight);
        }
    }
    return graph;
}

void test_assert(std::pair<std::vector<int>, double> result, std::string test_name, int expected_length, double expected_distance) {
    std::cout << test_name << " Length:" << result.first.size() << " Distance:" << result.second << std::endl;
    if (result.first.size() == expected_length && result.second == expected_distance) {
        std::cout << test_name << " PASSED" << std::endl;
    } else {
        std::cout << test_name << " FAILED" << std::endl;
    }
}

void tests(Graph* graph) {
    std::pair<std::vector<int>, double> result;
    auto itmo_node = graph->findClosestNode(30.30777, 59.95497, 0.001);
    auto home_node = graph->findClosestNode(30.382467, 59.979395, 0.001);//(30.374287, 59.980944, 0.001);//graph->findClosestNode(30.309108, 59.967238, 0.001);//graph->findClosestNode(30.374287, 59.980944, 0.001);

    //std::cout << "Src:" << home_node->id << " Dst:" << itmo_node->id << std::endl;
    result = graph->bfs_shortest(home_node, itmo_node);
    test_assert(result, "BFS", 118, 117);

    result = graph->bfs_shortest(home_node, home_node);
    test_assert(result, "BFS same node", 1, 0);

    result = graph->astar_shortest(home_node, itmo_node);
    test_assert(result, "A*", 140, 0.10066647269198108);

    result = graph->astar_shortest(home_node, home_node);
    test_assert(result, "A* same node", 1, 0);
    
    result = graph->dijkstra_shortest(home_node, itmo_node);
    test_assert(result, "Dijkstra", 140, 0.10066647269198108);

    result = graph->dijkstra_shortest(home_node, home_node);
    test_assert(result, "Dijkstra same node", 1, 0);

    result = graph->dfs(home_node, itmo_node);
    test_assert(result, "DFS", 8983, 8982);
    
    result = graph->dfs(home_node, itmo_node);
    test_assert(result, "DFS same node", 8983, 8982);
}

int main() {
    auto graph = load_graph_from_file("spb_graph.txt");

    tests(graph);

    return 0;
}

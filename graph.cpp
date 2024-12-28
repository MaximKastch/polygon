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

// Глобальные переменные для хранения временных меток
auto start = std::chrono::steady_clock::now(); 
    // std::chrono::steady_clock::time_point обычно ~8 байт (зависит от реализации).
    // O(1) по времени на инициализацию (runtime), O(1) по памяти
auto end = std::chrono::steady_clock::now();   
    // То же самое (~8 байт).
    // O(1) по времени, O(1) по памяти

// Функция запуска счётчика времени
void start_counter() {
    // Локальная операция присвоения: сама функция не выделяет память, 
    // перезаписывает значение глобальной переменной (start).
    // ~8 байт уже храним в глобальном объекте.
    // O(1) по времени, O(1) по памяти
    start = std::chrono::steady_clock::now(); 
}

// Функция остановки счётчика времени и вывода результата
void stop_counter(std::string algorithm) {
    // Пара локальных переменных:
    // 1) end - глобальная переменная (перезаписываем ~8 байт) -> O(1) по памяти
    // 2) elapsed - 64-битное целое (std::chrono::milliseconds::rep), обычно 8 байт -> O(1) по памяти
    // O(1) по времени: получение времени, вывод в поток
    end = std::chrono::steady_clock::now(); 
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); 
    // ~8 байт для счётчика
    // O(1) по времени, O(1) по памяти

    std::cout << algorithm <<" Execution time: " << elapsed << std::endl; 
    // Вывод в поток, временные строки в стеке
    // O(1) по времени, O(1) по памяти
}

// Функция splitString: разбивает строку по delimiter и возвращает vector<string>.
std::vector<std::string> splitString(const std::string& str, char delimiter) {
    // result - динамический массив std::vector<std::string>, начальное выделение обычно под маленькую ёмкость 
    // (например, 0 или 1-2 элементов), затем увеличение по мере добавления. 
    // Для каждого добавленного элемента выделяется память под std::string (обычно ~24-32 байта «на местах» + динамическая память под сам текст).
    // O(1) по памяти на создание самого объекта result (он пустой), далее - динамический рост
    // В худшем случае, если в строке будет k разделителей, результирующий вектор size() = k+1
    std::vector<std::string> result; 
    // O(1) по времени, O(1) по памяти

    std::stringstream ss(str); // std::stringstream внутри себя содержит буфер (динамическая память под копию str).
                               // O(L) по времени, где L = длина строки, чтобы скопировать внутрь (если реализовано так),
                               // O(L) по памяти под буфер
    // Локальная std::string (~24-32 байта под управляющие структуры + динамическая память при росте строки).
    // O(1) по времени (создание пустой строки), O(1) по памяти (управляющая структура)
    std::string item;          

    // Цикл читает из ss и кладёт во временный item. 
    // При push_back(item) в result создаются копии std::string. 
    // По времени: O(L) для прохода всей строки 
    // (каждый вызов std::getline занимает O(m) для выделенного участка).
    // По памяти: каждая новая std::string может потребовать O(m) динамической памяти (где m = длина сегмента).
    while (std::getline(ss, item, delimiter)) {
        // O(m) по времени на каждую итерацию при чтении сегмента,
        // O(1) амортизированно на вставку в vector (пока не превышена ёмкость).
        result.push_back(item); 
        // O(m) по памяти (содержимое строки) плюс ~24-32 байта на служебную часть std::string
    }

    return result; 
    // Возвращается по RVO/NRVO или move, доп. памяти не выделяется.
    // В целом асимптотика по времени: O(L) (где L = длина `str`),
    // по памяти: O(L) (размер под итоговые подстроки).
}

// Структура Node: хранит идентификатор, координаты и список соседей.
struct Node {
    int id;       // int обычно 4 байта.
                  // O(1) по памяти
    double lon;   // double обычно 8 байт.
                  // O(1) по памяти
    double lat;   // double обычно 8 байт.
                  // O(1) по памяти
    // Вектор пар (Node*, double). Каждый std::pair<Node*, double> обычно = 
    // (8 байт под Node*, 8 байт под double) = 16 байт на пару, плюс служебная память std::vector. 
    // Динамически растёт по мере добавления соседей.
    // O(1) при создании пустого, O(k) по памяти при наличии k соседей.
    std::vector<std::pair<Node*, double>> nodes; 
};

// Функтор сравнения для очереди с приоритетами
struct CompareNode {
    // Оператор(), принимает две пары (double, Node*) и сравнивает first. 
    // Сам функтор не хранит полей, значит не занимает память на уровне экземпляра, 
    // только в момент вызова идёт сравнение.
    // O(1) по времени на каждое сравнение, O(1) по памяти
    bool operator()(const std::pair<double, Node*>& a, const std::pair<double, Node*>& b) const {
        return a.first > b.first; // O(1) сравнение
    }
};

// Структура Graph: хранит узлы в хеш-таблице и реализует алгоритмы поиска путей.
struct Graph {
    // Основное поле — std::unordered_map<std::string, Node*>
    // - для каждой пары (ключ-значение) хранится ~ (хэш + key + value) в хеш-таблице.
    // - ключ (std::string) может занимать динамическую память (зависит от длины строки).
    // - value (Node*) — 8 байт (указатель).
    // О(1) средняя вставка/поиск, О(N) в худшем случае (коллизии).
    // По памяти: O(N * (size of string + 8 байт на Node*)) + служебные поля
    std::unordered_map<std::string, Node*> nodes;
    
    int max_id_current = 0; // 4 байта (int).
                            // O(1) по памяти

    // Создаём или возвращаем существующий узел по строке координат
    Node* getOrAddNodeByCoordinatesLine(std::string coordinates_string) {
        // Локальная переменная it — это итератор std::unordered_map, занимает не более ~8-16 байт на стеке.
        // Поиск в хеш-таблице: O(1) в среднем, O(N) в худшем
        auto it = this->nodes.find(coordinates_string); 
        if (it != this->nodes.end()) {
            return it->second; 
            // Если нашли, не выделяем ничего нового
            // O(1) по времени, O(1) по памяти
        }

        // Создаём новый узел (Node).
        // Выделение памяти в куче под сам Node. 
        // O(1) по времени (амортизированно), O(1) по памяти (фиксированный размер Node)
        Node* current_node = new Node(); 

        // splitString вернёт вектор строк; 
        // в зависимости от длины входной строки будет несколько std::string в vector.
        // O(L) по времени, O(L) по памяти
        std::vector<std::string> current_coordinates = splitString(coordinates_string, ',');

        current_node->id = this->max_id_current;   
        // O(1) по времени, O(1) по памяти (id уже внутри Node)
        current_node->lon = std::stod(current_coordinates[0]); 
        // std::stod может аллоцировать временные объекты в стеке, O(1) по времени для короткой строки
        current_node->lat = std::stod(current_coordinates[1]); 
        current_node->nodes = std::vector<std::pair<Node*, double>>(); 
            // Пустой вектор (минимальное внутреннее хранилище, обычно 0..16 байт).
            // O(1) по времени, O(1) по памяти

        // Запись в хеш-таблицу: хеш-таблица может перераспределяться (re-hash), 
        // выделяя память под новые бакеты. 
        // В среднем O(1) по времени, O(N) в худшем.
        // По памяти: храним key (string) + value (Node*) + служебный узел.
        this->nodes[coordinates_string] = current_node; 
        this->max_id_current++; 
        // инкремент (4 байта int).
        // O(1) по времени, O(1) по памяти

        return current_node; 
        // O(1) по времени, O(1) по памяти
    }

    // Находит среди всех узлов тот, который ближе всего к (lon, lat)
    Node* findClosestNode(double lon, double lat, 
                          double min_distance = std::numeric_limits<double>::max()) {
        // node_founded — локальный указатель (8 байт на стеке).
        // O(1) по памяти
        Node* node_founded = nullptr; 

        // Проходим по всей хеш-таблице (nodes). 
        // O(N) по времени, где N — кол-во узлов
        // Нет новых крупных аллокаций, только чтение данных из map.
        for (const auto &current : nodes) {
            // O(1) извлечение second
            Node* node = current.second; // 8-байтовый указатель
            double distance = std::sqrt(std::pow(node->lat - lat, 2) + 
                                        std::pow(node->lon - lon, 2)); 
            // O(1) по времени вычисление
            // Локальная double переменная distance (~8 байт в стеке).

            if (distance < min_distance) {
                node_founded = node;  
                min_distance = distance; 
                // O(1)
            }
        }
        return node_founded; 
        // O(1) по памяти
    }

    // Вспомогательный метод для вычисления евклидова расстояния
    double distance_between(const Node* a, const Node* b) {
        // Возвращает double (~8 байт), вычисляет sqrt(...).
        // O(1) по времени, O(1) по памяти
        return std::sqrt(std::pow(a->lon - b->lon, 2) + std::pow(a->lat - b->lat, 2));
    }

    // Восстановление пути по map `came_from`
    // Возвращает вектор int (id), динамически аллоцированный внутри std::vector.
    std::vector<int> reconstructPath(
        const std::unordered_map<int, int>& came_from,
        const int& start_node_id, const int& goal_node_id) 
    {
        // path - вектор int; каждый int ~4 байта. Размер зависит от длины пути.
        // O(k) по памяти, где k — длина пути
        std::vector<int> path; 
        // Цикл идёт от goal_node_id назад к start_node_id, 
        // используя came_from (хеш-таблица). В среднем доступ O(1), худший O(k).
        // В худшем случае k ~ N, если путь проходит через все узлы.
        for (int at = goal_node_id; 
             at != start_node_id; 
             at = came_from.at(at)) 
        {
            path.push_back(at); // O(1) амортизированно при вставке в вектор
        }
        path.push_back(start_node_id); 
        // O(1)

        // Реверс вручную
        std::vector<int> reversed_path(path.size()); 
        // Выделение массива size() * 4 байта под int. O(k) по памяти
        for (size_t i = 0, j = path.size() - 1; i < path.size(); ++i, --j) {
            reversed_path[i] = path[j]; 
            // O(1) в каждой итерации
        }
        return reversed_path; 
        // RVO/мув
        // Итог: O(k) по времени и памяти, где k — длина пути
    }

    // Возвращает g_cost[node_id], если он есть, иначе бесконечность
    double getGCost(const std::unordered_map<int, double>& g_cost,
                    const int& node_id)
    {
        // В среднем O(1) доступ, в худшем O(V) (коллизии)
        // Возвращается double (8 байт).
        auto it = g_cost.find(node_id);
        if (it == g_cost.end()) {
            return std::numeric_limits<double>::infinity();
        }
        return it->second;
    }

    //---------------------------------------------------------//
    // 1) BFS — Возвращаем (путь, стоимость)
    //    Стоимость = кол-во рёбер (граф невзвешенный).
    // По времени - O(V + E)
    // По памяти - O(V
    //---------------------------------------------------------//
    std::pair<std::vector<int>, double> bfs_shortest(Node* start_node, Node* goal_node) 
    {
        // Стартуем счётчик времени
        // O(1)
        start_counter();

        // Если старт и цель — один и тот же узел
        if (start_node->id == goal_node->id) {
            stop_counter("BFS");
            // Возвращаем путь, состоящий из одного узла, и стоимость 0
            // O(1)
            return {{start_node->id}, 0}; 
        }

        // Подготовка структур:
        // queue<Node*>    — динамическая структура, хранит указатели на Node. 8 байт на указатель Node + 40-100 байт сама очередь
        // came_from       — хеш-таблица int->int. 24-32 байта на каждый добавленный элемент, включая служебную информацию + 64 - 80 байт на пустую хэш таблицу
        // distances       — хеш-таблица int->double. 24-32 байта на каждый добавленный элемент, включая служебную информацию + 64 - 80 байт на пустую хэш таблицу
        // visited         — хеш-сет int. 16-24 байта на каждый добавленный элемент, включая служебную информацию + 64 - 80 байт на пустую хэш таблицу
        
        // По времени: Все операции с хеш-таблицами и очередью — в среднем O(1) на вставку/извлечение, 
        // BFS в целом O(V + E).
        // По памяти: O(V) на visited, came_from, distances; O(V) на queue в худшем случае.
        std::queue<Node*> queue;            
        std::unordered_map<int, int> came_from; 
        std::unordered_map<int, double> distances; 
        std::unordered_set<int> visited;          

        // Присвоение в таблицу и множество
        // O(1) в среднем
        distances[start_node->id] = 0.0; 
        visited.insert(start_node->id); 
        queue.push(start_node); 


        // Основной цикл BFS
        while (!queue.empty()) {
            Node* current_node = queue.front(); // 8 байт
            // O(1)
            queue.pop(); 
            // O(1)

            if (current_node->id == goal_node->id) {
                double cost = distances[goal_node->id]; // 8 байт
                // Достаём double из map, 8 байт
                auto path = reconstructPath(came_from, start_node->id, goal_node->id); 
                // path — std::vector<int>, 4 байта на int + 20-32 байт сам вектор
                stop_counter("BFS");
                return { path, cost };
            }

            // Перебираем всех соседей current_node
            // current_node->nodes: std::vector<std::pair<Node*, double>>
            // Каждый push может аллоцировать память в queue. 
            // Суммарно по всем узлам: O(V + E) для обхода.
            for (const auto& neighbor : current_node->nodes) {
                Node* neighbor_node = neighbor.first; // 8 байт на указатель Node
                if (visited.find(neighbor_node->id) == visited.end()) {
                    // O(1) в среднем
                    visited.insert(neighbor_node->id);          
                    came_from[neighbor_node->id] = current_node->id; 
                    distances[neighbor_node->id] = distances[current_node->id] + 1.0; 
                    queue.push(neighbor_node); 
                }
            }
        }
        stop_counter("BFS");
        // Если путь не найден
        // O(1)
        return {{}, std::numeric_limits<double>::infinity()}; 
    }

    //---------------------------------------------------------//
    // 1) DFS — Возвращаем (путь, стоимость)
    //    Стоимость = кол-во рёбер (граф невзвешенный).
    // По времени - O(V + E)
    // По памяти - O(V)
    //---------------------------------------------------------//
    std::pair<std::vector<int>, double> dfs(Node* start_node, Node* goal_node)
    {
        start_counter();
        // O(1)
        if (start_node->id == goal_node->id) {
            stop_counter("DFS");
            // O(1)
            return {{start_node->id}, 0}; 
        }
        
        // Стек для итеративного DFS
        // Хранит указатели на Node 8 байт на каждый элемент + 40-100 байт на сам контейнер (std::deque) внутри stack.
        // O(V) потенциально в худшем случае
        std::stack<const Node*> st;
        // Набор (hash set) по id (int). 16-24 байт на элемент, включая служебную информацию + 64-80 на сам set
        // O(V) в худшем случае
        std::unordered_set<int> visited;
        // Карта предков (int->int). 24-32 байт на элемент, включая служебную информацию + 64-80 на сам map
        // O(V) в худшем случае
        std::unordered_map<int, int> came_from;

        // Инициализация
        // O(1)
        st.push(start_node);
        visited.insert(start_node->id);

        // По времени: в худшем случае DFS O(V + E).
        while (!st.empty()) {
            const Node* current = st.top(); // 8 байт
            st.pop();

            if (current == goal_node) {
                // Восстанавливаем путь
                // O(k), где k — длина пути
                std::vector<int> path = reconstructPath(came_from, start_node->id, goal_node->id);
                // path — std::vector<int>, 4 байта на int + 20-32 байт сам вектор
                double edge_count = static_cast<double>(path.size() - 1); // 8 байт
                stop_counter("DFS");
                return { path, edge_count };
            }

            // Смотрим всех соседей
            // O(deg(current)) за одну итерацию; суммарно O(E) для всего обхода
            for (auto& edge : current->nodes) {
                const Node* neighbor = edge.first;  // 8 байт на указатель Node
                if (!neighbor) continue; 
                if (visited.find(neighbor->id) == visited.end()) {
                    visited.insert(neighbor->id);
                    came_from[neighbor->id] = current->id;
                    st.push(neighbor);
                }
            }
        }

        stop_counter("DFS");
        // Путь не найден
        // O(1)
        return {{}, 0.0};
    }

    //---------------------------------------------------------//
    // 3) Ленивая Dijkstra — (путь, суммарный вес)
    // По времени - O(E * log V) в худшем случае
    // По памяти - O(V)
    //---------------------------------------------------------//
    std::pair<std::vector<int>, double> dijkstra_shortest(Node* start_node, Node* goal_node)
    {
        start_counter();
        // O(1)
        if (start_node->id == goal_node->id) {
            stop_counter("Dijkstra");
            return {{start_node->id}, 0};
        }

        // Приоритетная очередь (мин-куча) хранит пары (double, Node*)
        // - внутри себя vector<std::pair<double, Node*>> 
        // - каждое помещение ~O(log V) в худшем случае
        // - хранение ~ (16 байт на элемент = 8 double + 8 указатель на Node) + 20-40 байт сама очередь
        std::priority_queue<
            std::pair<double, Node*>,
            std::vector<std::pair<double, Node*>>,
            CompareNode
        > pq;

        // distances — hash map: int -> double. 24-32 на элемент с учетом служебной информации + 64-80 на сам map
        // came_from — hash map: int -> int. 24-32 на элемент с учетом служебной информации +  64-80 на сам map
        // По времени: Dijkstra — O(E log V) в худшем случае (при «ленивой» проверке).
        // По памяти: O(V) на хранение distance, came_from, + O(V) в приоритетной очереди в худшем случае.
        std::unordered_map<int, double> distances;
        std::unordered_map<int, int> came_from;

        distances[start_node->id] = 0.0; 
        pq.emplace(0.0, start_node); 

        // Dijkstra: O(E log V) в худшем случае
        while (!pq.empty()) {
            double current_dist = pq.top().first; // 8 байт
            Node* current_node  = pq.top().second; // 8 байт на указатель Node
            pq.pop();           

            double best_known_for_current = getGCost(distances, current_node->id); // 8 байт
            // O(1) среднее

            if (current_dist > best_known_for_current) {
                continue; 
                // Ленивая проверка
                // O(1)
            }

            if (current_node == goal_node) {
                std::vector<int> path = reconstructPath(came_from, start_node->id, goal_node->id);
                // path — std::vector<int>, 4 байта на int + 20-32 байт сам вектор, O(k)
                double cost = getGCost(distances, goal_node->id); // 8 байт
                // O(1)
                stop_counter("Dijkstra");
                return { path, cost };
            }

            // Перебираем всех соседей
            // Суммарно за всё время: O(E) итераций
            for (auto& edge : current_node->nodes) {
                Node* neighbor_node = edge.first; // 8 байт
                double edge_weight  = edge.second; // 8 байт

                double old_dist = getGCost(distances, neighbor_node->id); // 8 байт
                // O(1) среднее
                double new_dist = current_dist + edge_weight; // 8 байт

                if (new_dist < old_dist) {
                    distances[neighbor_node->id] = new_dist;
                    came_from[neighbor_node->id] = current_node->id;
                    pq.emplace(new_dist, neighbor_node); 
                    // Вставка в очередь ~O(log V)
                }
            }
        }

        stop_counter("Dijkstra");
        // Путь не найден
        return {{}, std::numeric_limits<double>::infinity()};
    }

    //---------------------------------------------------------//
    // 4) A* (A-Star) — (путь, суммарный вес)
    // По времени - O(E*log V) в худшем случае
    // По памяти - O(V)
    //---------------------------------------------------------//
    std::pair<std::vector<int>, double> astar_shortest(Node* start_node, Node* goal_node)
    {
        start_counter();
        // O(1)
        if (start_node->id == goal_node->id) {
            stop_counter("A*");
            return {{start_node->id}, 0};
        }

        // open_set — приоритетная очередь (fCost = gCost + heuristic).
        // O(log V) на вставку/извлечение
        std::priority_queue<
            std::pair<double, Node*>,
            std::vector<std::pair<double, Node*>>,
            CompareNode
        > open_set;

        // g_cost, f_cost — hash map: int -> double. 24-32 на элемент с учетом служебной информации + 64-80 на сам map
        // came_from — hash map: int -> int. 24-32 на элемент с учетом служебной информации +  64-80 на сам map
        // A* ~ O(E log V) в худшем случае
        std::unordered_map<int, double> g_cost; 
        std::unordered_map<int, double> f_cost; 
        std::unordered_map<int, int> came_from;

        g_cost[start_node->id] = 0.0; 
        f_cost[start_node->id] = distance_between(start_node, goal_node); 
        open_set.emplace(f_cost[start_node->id], start_node); 

        // A* ~ O(E log V) в худшем случае, зависит от эвристики
        while (!open_set.empty()) {
            Node* current = open_set.top().second; // 8 байт
            open_set.pop(); 

            if (current == goal_node) {
                auto path = reconstructPath(came_from, start_node->id, goal_node->id);
                // path — std::vector<int>, 4 байта на int + 20-32 байт сам вектор, O(k)
                double cost = g_cost[goal_node->id]; // 8 байт O(1)
                stop_counter("A*");
                return { path, cost };
            }

            // Перебираем все ребра из current
            // Суммарно за все итерации O(E)
            for (auto& edge : current->nodes) {
                Node* neighbor_node = edge.first; // 8 байт 
                double edge_weight = edge.second; // 8 байт 

                double current_g = getGCost(g_cost, current->id); // 8 байт 
                double tentative_g_cost = current_g + edge_weight; // 8 байт 
                double neighbor_g = getGCost(g_cost, neighbor_node->id); // 8 байт 

                if (tentative_g_cost < neighbor_g) {
                    g_cost[neighbor_node->id] = tentative_g_cost;
                    double heuristic = distance_between(neighbor_node, goal_node); // 8 байт 
                    f_cost[neighbor_node->id] = tentative_g_cost + heuristic;
                    came_from[neighbor_node->id] = current->id;

                    open_set.emplace(f_cost[neighbor_node->id], neighbor_node);
                    // ~O(log V)
                }
            }
        }

        stop_counter("A*");
        // Если путь не найден
        // O(1)
        return {{}, std::numeric_limits<double>::infinity()};
    }
};

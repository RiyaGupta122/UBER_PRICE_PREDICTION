/*
 * DSA Algorithms for Uber Price Optimization
 * C implementation of core data structures and algorithms
 * Compiled as shared library for Python integration
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>

#define MAX_NODES 1000
#define MAX_DRIVERS 500
#define MAX_RIDERS 500
#define INFINITY 999999

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef struct {
    double lat;
    double lng;
    char name[100];
} Location;

typedef struct {
    int id;
    Location location;
    double rating;
    bool is_available;
    int completed_rides;
    char vehicle_type[20];
} Driver;

typedef struct {
    int id;
    Location pickup;
    Location dropoff;
    double max_price;
    char preferences[50];
} Rider;

typedef struct {
    Driver driver;
    double priority;
} HeapNode;

typedef struct {
    HeapNode* heap;
    int size;
    int capacity;
} PriorityQueue;

typedef struct {
    int** adj_matrix;
    double** weights;
    int num_nodes;
} Graph;

// ============================================================================
// HAVERSINE DISTANCE CALCULATION (Optimized C Implementation)
// ============================================================================

double to_radians(double degrees) {
    return degrees * M_PI / 180.0;
}

double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    const double R = 6371.0; // Earth's radius in kilometers
    
    double lat1_rad = to_radians(lat1);
    double lon1_rad = to_radians(lon1);
    double lat2_rad = to_radians(lat2);
    double lon2_rad = to_radians(lon2);
    
    double dlat = lat2_rad - lat1_rad;
    double dlon = lon2_rad - lon1_rad;
    
    double a = sin(dlat / 2) * sin(dlat / 2) + 
               cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) * sin(dlon / 2);
    
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    
    return R * c;
}

// ============================================================================
// PRIORITY QUEUE (MIN-HEAP) FOR DRIVER ALLOCATION
// ============================================================================

PriorityQueue* create_priority_queue(int capacity) {
    PriorityQueue* pq = (PriorityQueue*)malloc(sizeof(PriorityQueue));
    pq->heap = (HeapNode*)malloc(capacity * sizeof(HeapNode));
    pq->size = 0;
    pq->capacity = capacity;
    return pq;
}

void swap_heap_nodes(HeapNode* a, HeapNode* b) {
    HeapNode temp = *a;
    *a = *b;
    *b = temp;
}

void heapify_up(PriorityQueue* pq, int index) {
    if (index == 0) return;
    
    int parent = (index - 1) / 2;
    if (pq->heap[parent].priority > pq->heap[index].priority) {
        swap_heap_nodes(&pq->heap[parent], &pq->heap[index]);
        heapify_up(pq, parent);
    }
}

void heapify_down(PriorityQueue* pq, int index) {
    int left = 2 * index + 1;
    int right = 2 * index + 2;
    int smallest = index;
    
    if (left < pq->size && pq->heap[left].priority < pq->heap[smallest].priority) {
        smallest = left;
    }
    
    if (right < pq->size && pq->heap[right].priority < pq->heap[smallest].priority) {
        smallest = right;
    }
    
    if (smallest != index) {
        swap_heap_nodes(&pq->heap[index], &pq->heap[smallest]);
        heapify_down(pq, smallest);
    }
}

double calculate_driver_priority(Driver* driver, Location* pickup_location) {
    if (!driver->is_available) return INFINITY;
    
    // Calculate distance to pickup
    double distance = haversine_distance(
        driver->location.lat, driver->location.lng,
        pickup_location->lat, pickup_location->lng
    );
    
    // Priority factors (lower value = higher priority)
    double distance_factor = distance;
    double rating_factor = (5.0 - driver->rating) * 0.5;
    double experience_factor = fmax(0, (100 - driver->completed_rides) * 0.01);
    
    return distance_factor + rating_factor + experience_factor;
}

void insert_driver(PriorityQueue* pq, Driver* driver, Location* pickup_location) {
    if (pq->size >= pq->capacity) return;
    
    HeapNode node;
    node.driver = *driver;
    node.priority = calculate_driver_priority(driver, pickup_location);
    
    pq->heap[pq->size] = node;
    heapify_up(pq, pq->size);
    pq->size++;
}

Driver* get_nearest_driver(PriorityQueue* pq) {
    if (pq->size == 0) return NULL;
    
    Driver* nearest = (Driver*)malloc(sizeof(Driver));
    *nearest = pq->heap[0].driver;
    
    // Move last element to root and heapify down
    pq->heap[0] = pq->heap[pq->size - 1];
    pq->size--;
    
    if (pq->size > 0) {
        heapify_down(pq, 0);
    }
    
    return nearest;
}

// ============================================================================
// DIJKSTRA'S ALGORITHM FOR SHORTEST PATH
// ============================================================================

Graph* create_graph(int num_nodes) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->num_nodes = num_nodes;
    
    // Allocate adjacency matrix
    graph->adj_matrix = (int**)malloc(num_nodes * sizeof(int*));
    graph->weights = (double**)malloc(num_nodes * sizeof(double*));
    
    for (int i = 0; i < num_nodes; i++) {
        graph->adj_matrix[i] = (int*)calloc(num_nodes, sizeof(int));
        graph->weights[i] = (double*)malloc(num_nodes * sizeof(double));
        
        for (int j = 0; j < num_nodes; j++) {
            graph->weights[i][j] = INFINITY;
        }
    }
    
    return graph;
}

void add_edge(Graph* graph, int src, int dest, double weight) {
    graph->adj_matrix[src][dest] = 1;
    graph->weights[src][dest] = weight;
    
    // For undirected graph
    graph->adj_matrix[dest][src] = 1;
    graph->weights[dest][src] = weight;
}

int find_min_distance_vertex(double* distances, bool* visited, int num_nodes) {
    double min_dist = INFINITY;
    int min_vertex = -1;
    
    for (int v = 0; v < num_nodes; v++) {
        if (!visited[v] && distances[v] < min_dist) {
            min_dist = distances[v];
            min_vertex = v;
        }
    }
    
    return min_vertex;
}

double* dijkstra_shortest_path(Graph* graph, int start, int end, int* path_length) {
    double* distances = (double*)malloc(graph->num_nodes * sizeof(double));
    bool* visited = (bool*)calloc(graph->num_nodes, sizeof(bool));
    int* previous = (int*)malloc(graph->num_nodes * sizeof(int));
    
    // Initialize distances
    for (int i = 0; i < graph->num_nodes; i++) {
        distances[i] = INFINITY;
        previous[i] = -1;
    }
    distances[start] = 0;
    
    // Dijkstra's algorithm
    for (int count = 0; count < graph->num_nodes - 1; count++) {
        int u = find_min_distance_vertex(distances, visited, graph->num_nodes);
        if (u == -1) break;
        
        visited[u] = true;
        
        // Update distances of adjacent vertices
        for (int v = 0; v < graph->num_nodes; v++) {
            if (!visited[v] && graph->adj_matrix[u][v] && 
                distances[u] != INFINITY && 
                distances[u] + graph->weights[u][v] < distances[v]) {
                
                distances[v] = distances[u] + graph->weights[u][v];
                previous[v] = u;
            }
        }
    }
    
    // Reconstruct path
    int path_count = 0;
    int current = end;
    while (current != -1) {
        path_count++;
        current = previous[current];
    }
    
    *path_length = path_count;
    
    free(visited);
    free(previous);
    
    return distances;
}

// ============================================================================
// BIPARTITE MATCHING (HUNGARIAN ALGORITHM SIMPLIFIED)
// ============================================================================

typedef struct {
    int rider_id;
    int driver_id;
    double cost;
} Assignment;

double calculate_matching_cost(Rider* rider, Driver* driver) {
    if (!driver->is_available) return INFINITY;
    
    // Distance cost
    double distance = haversine_distance(
        rider->pickup.lat, rider->pickup.lng,
        driver->location.lat, driver->location.lng
    );
    
    // Wait time estimation (simplified)
    double wait_time = distance / 25.0 * 60; // Assuming 25 km/h average speed
    
    // Rating factor
    double rating_factor = (5.0 - driver->rating) * 2.0;
    
    // Weighted cost calculation
    double total_cost = distance * 0.4 + wait_time * 0.3 + rating_factor * 0.3;
    
    return total_cost;
}

Assignment* find_optimal_assignments(Rider* riders, int num_riders, 
                                   Driver* drivers, int num_drivers, 
                                   int* num_assignments) {
    Assignment* assignments = (Assignment*)malloc(num_riders * sizeof(Assignment));
    bool* driver_assigned = (bool*)calloc(num_drivers, sizeof(bool));
    int assignment_count = 0;
    
    // Simplified greedy assignment (for demonstration)
    // In production, would implement full Hungarian algorithm
    for (int r = 0; r < num_riders; r++) {
        int best_driver = -1;
        double min_cost = INFINITY;
        
        for (int d = 0; d < num_drivers; d++) {
            if (!driver_assigned[d]) {
                double cost = calculate_matching_cost(&riders[r], &drivers[d]);
                if (cost < min_cost) {
                    min_cost = cost;
                    best_driver = d;
                }
            }
        }
        
        if (best_driver != -1) {
            assignments[assignment_count].rider_id = riders[r].id;
            assignments[assignment_count].driver_id = drivers[best_driver].id;
            assignments[assignment_count].cost = min_cost;
            driver_assigned[best_driver] = true;
            assignment_count++;
        }
    }
    
    *num_assignments = assignment_count;
    free(driver_assigned);
    
    return assignments;
}

// ============================================================================
// DYNAMIC PRICING ALGORITHM
// ============================================================================

typedef struct {
    double base_price;
    double surge_multiplier;
    double demand_factor;
    double supply_factor;
    double final_price;
} PricingResult;

PricingResult calculate_dynamic_pricing(double base_price, double demand_level, 
                                      double supply_level, double weather_factor, 
                                      double traffic_factor, double time_factor) {
    PricingResult result;
    result.base_price = base_price;
    
    // Supply-demand ratio
    double supply_demand_ratio = demand_level / supply_level;
    double surge_multiplier = fmax(1.0, supply_demand_ratio);
    
    // Apply exponential surge for high demand
    if (surge_multiplier > 2.0) {
        surge_multiplier = 2.0 + pow(surge_multiplier - 2.0, 1.5);
    }
    
    // Environmental factors
    double environmental_multiplier = weather_factor * time_factor * traffic_factor;
    
    // Final price calculation
    double final_price = base_price * surge_multiplier * environmental_multiplier;
    
    // Price elasticity adjustment (avoid extreme prices)
    double max_surge = 5.0;
    final_price = fmin(final_price, base_price * max_surge);
    
    result.surge_multiplier = fmin(surge_multiplier, max_surge);
    result.demand_factor = demand_level;
    result.supply_factor = supply_level;
    result.final_price = final_price;
    
    return result;
}

// ============================================================================
// MAIN TESTING FUNCTION
// ============================================================================

void test_dsa_algorithms() {
    printf("🔧 Testing DSA Algorithms for Uber Optimization\n");
    printf("================================================\n");
    
    // Test Haversine distance
    double dist = haversine_distance(19.0760, 72.8777, 18.9690, 72.8205);
    printf("Distance Mumbai to Mumbai Central: %.2f km\n", dist);
    
    // Test Priority Queue
    PriorityQueue* pq = create_priority_queue(10);
    Location pickup = {19.0760, 72.8777, "Mumbai"};
    
    Driver drivers[3] = {
        {1, {19.0800, 72.8800, "Driver1"}, 4.5, true, 150, "UberGo"},
        {2, {19.0700, 72.8700, "Driver2"}, 4.8, true, 200, "UberGo"},
        {3, {19.0900, 72.8900, "Driver3"}, 4.2, true, 100, "UberGo"}
    };
    
    for (int i = 0; i < 3; i++) {
        insert_driver(pq, &drivers[i], &pickup);
    }
    
    Driver* nearest = get_nearest_driver(pq);
    if (nearest) {
        printf("Nearest driver: ID %d, Rating %.1f\n", nearest->id, nearest->rating);
        free(nearest);
    }
    
    // Test Dynamic Pricing
    PricingResult pricing = calculate_dynamic_pricing(100.0, 2.5, 1.2, 1.3, 1.5, 1.2);
    printf("Dynamic Pricing: Base ₹%.2f -> Final ₹%.2f (%.1fx surge)\n", 
           pricing.base_price, pricing.final_price, pricing.surge_multiplier);
    
    printf("================================================\n");
    
    free(pq->heap);
    free(pq);
}

int main() {
    test_dsa_algorithms();
    return 0;
}
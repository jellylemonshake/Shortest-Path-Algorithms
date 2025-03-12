#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define MAX_VERTICES 20
#define INF INT_MAX // Define infinity

// Structure for heap nodes used in priority queue implementations
typedef struct {
    int vertex;
    int distance;
} HeapNode;

// Structure for min heap used as priority queue
typedef struct {
    HeapNode* array;
    int size;
    int capacity;
    int* position; // To track positions of vertices in heap
} MinHeap;

// Structure for storing edge information
typedef struct {
    int src, dest, weight;
} Edge;

// For A* algorithm: storing node information
typedef struct {
    int vertex;
    int g_score; // Cost from start to current
    int f_score; // Estimated total cost (g_score + heuristic)
    int parent;
} AStarNode;

// Function to create a new min heap
MinHeap* createMinHeap(int capacity) {
    MinHeap* minHeap = (MinHeap*)malloc(sizeof(MinHeap));
    minHeap->position = (int*)malloc(capacity * sizeof(int));
    minHeap->size = 0;
    minHeap->capacity = capacity;
    minHeap->array = (HeapNode*)malloc(capacity * sizeof(HeapNode));
    return minHeap;
}

// Utility function to swap two heap nodes
void swapHeapNode(HeapNode* a, HeapNode* b) {
    HeapNode temp = *a;
    *a = *b;
    *b = temp;
}

// Standard min heapify function
void minHeapify(MinHeap* minHeap, int idx) {
    int smallest, left, right;
    smallest = idx;
    left = 2 * idx + 1;
    right = 2 * idx + 2;

    if (left < minHeap->size &&
        minHeap->array[left].distance < minHeap->array[smallest].distance)
        smallest = left;

    if (right < minHeap->size &&
        minHeap->array[right].distance < minHeap->array[smallest].distance)
        smallest = right;

    if (smallest != idx) {
        // Update positions in position array
        minHeap->position[minHeap->array[smallest].vertex] = idx;
        minHeap->position[minHeap->array[idx].vertex] = smallest;

        // Swap nodes
        swapHeapNode(&minHeap->array[smallest], &minHeap->array[idx]);
        minHeapify(minHeap, smallest);
    }
}

// Check if min heap is empty
int isEmpty(MinHeap* minHeap) {
    return minHeap->size == 0;
}

// Extract minimum node from heap
HeapNode extractMin(MinHeap* minHeap) {
    if (isEmpty(minHeap))
        return (HeapNode){-1, INT_MAX};

    // Store the root node
    HeapNode root = minHeap->array[0];

    // Replace root with last node
    HeapNode lastNode = minHeap->array[minHeap->size - 1];
    minHeap->array[0] = lastNode;

    // Update position of the last node
    minHeap->position[root.vertex] = minHeap->size - 1;
    minHeap->position[lastNode.vertex] = 0;

    // Reduce heap size and heapify root
    --minHeap->size;
    minHeapify(minHeap, 0);

    return root;
}

// Decrease key value of a given vertex
void decreaseKey(MinHeap* minHeap, int vertex, int distance) {
    // Get the index of vertex in heap array
    int i = minHeap->position[vertex];

    // Update distance value
    minHeap->array[i].distance = distance;

    // Travel up while complete heap property is not satisfied
    while (i && minHeap->array[i].distance < minHeap->array[(i - 1) / 2].distance) {
        // Swap with parent
        minHeap->position[minHeap->array[i].vertex] = (i - 1) / 2;
        minHeap->position[minHeap->array[(i - 1) / 2].vertex] = i;
        swapHeapNode(&minHeap->array[i], &minHeap->array[(i - 1) / 2]);

        // Move to parent index
        i = (i - 1) / 2;
    }
}

// Check if a vertex is in the min heap
bool isInMinHeap(MinHeap* minHeap, int vertex) {
    return minHeap->position[vertex] < minHeap->size;
}

// Function to print the solution
void printSolution(int dist[], int parent[], int src, int V, const char* algorithm) {
    printf("\n%s Algorithm Results:\n", algorithm);
    printf("+---------+------------------------+--------------------------------+\n");
    printf("| Vertex  | Distance from Source   | Path                           |\n");
    printf("+---------+------------------------+--------------------------------+\n");
   
    for (int i = 0; i < V; i++) {
        printf("| %-7d | ", i);
       
        // Print distance
        if (dist[i] == INF)
            printf("%-22s | ", "INF");
        else
            printf("%-22d | ", dist[i]);
       
        // Print the path
        if (dist[i] == INF) {
            printf("%-30s |\n", "No path");
        } else if (i == src) {
            printf("%-30d |\n", i);
        } else {
            // Store the path in an array
            int path[MAX_VERTICES];
            int pathLen = 0;
            int current = i;
           
            while (current != -1) {
                path[pathLen++] = current;
                current = parent[current];
            }
           
            // Print path in reverse order (from source to destination)
            char pathStr[100] = "";
            int offset = 0;
            for (int j = pathLen - 1; j >= 0; j--) {
                offset += sprintf(pathStr + offset, "%d", path[j]);
                if (j > 0) offset += sprintf(pathStr + offset, " -> ");
            }
           
            printf("%-30s |\n", pathStr);
        }
    }
    printf("+---------+------------------------+--------------------------------+\n");
}

// Print all-pairs shortest paths (used by Floyd-Warshall)
void printAllPairsSolution(int dist[MAX_VERTICES][MAX_VERTICES], int path[MAX_VERTICES][MAX_VERTICES], int V) {
    printf("\nFloyd-Warshall Algorithm Results (All-Pairs Shortest Paths):\n");
   
    for (int src = 0; src < V; src++) {
        printf("\nShortest paths from vertex %d:\n", src);
        printf("+---------+------------------------+--------------------------------+\n");
        printf("| To      | Distance               | Path                           |\n");
        printf("+---------+------------------------+--------------------------------+\n");
       
        for (int dest = 0; dest < V; dest++) {
            if (src == dest) continue; // Skip self-loops
           
            printf("| %-7d | ", dest);
           
            // Print distance
            if (dist[src][dest] == INF)
                printf("%-22s | ", "INF");
            else
                printf("%-22d | ", dist[src][dest]);
           
            // Print the path
            if (dist[src][dest] == INF) {
                printf("%-30s |\n", "No path");
            } else {
                // Reconstruct path using the path matrix
                int intermediate[MAX_VERTICES];
                int pathLen = 0;
               
                // Function to recursively find the path
                void findPath(int s, int d) {
                    if (path[s][d] == -1) {
                        intermediate[pathLen++] = s;
                        if (s != d) intermediate[pathLen++] = d;
                        return;
                    }
                   
                    findPath(s, path[s][d]);
                    pathLen--; // Remove duplicate vertex
                    findPath(path[s][d], d);
                }
               
                findPath(src, dest);
               
                // Print the path
                char pathStr[100] = "";
                int offset = 0;
                for (int j = 0; j < pathLen; j++) {
                    offset += sprintf(pathStr + offset, "%d", intermediate[j]);
                    if (j < pathLen - 1) offset += sprintf(pathStr + offset, " -> ");
                }
               
                printf("%-30s |\n", pathStr);
            }
        }
        printf("+---------+------------------------+--------------------------------+\n");
    }
}

// Function to find the vertex with minimum distance value
int minDistance(int dist[], bool sptSet[], int V) {
    int min = INF, min_index;
   
    for (int v = 0; v < V; v++) {
        if (sptSet[v] == false && dist[v] < min) {
            min = dist[v];
            min_index = v;
        }
    }
   
    return min_index;
}

// Implementation of Dijkstra's algorithm
void dijkstra(int graph[MAX_VERTICES][MAX_VERTICES], int src, int V) {
    int dist[MAX_VERTICES];     // Distance from source to each vertex
    bool sptSet[MAX_VERTICES];  // sptSet[i] will be true if vertex i is included in shortest path tree
    int parent[MAX_VERTICES];   // Parent array to store shortest path tree
   
    // Initialize all distances as INFINITE and sptSet[] as false
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
        sptSet[i] = false;
        parent[i] = -1;
    }
   
    // Distance of source vertex from itself is always 0
    dist[src] = 0;
   
    // Step by step results for Dijkstra's algorithm
    printf("\nDijkstra's Algorithm Steps:\n");
    printf("+--------+------------------------------------------------------------+\n");
   
    // Find shortest path for all vertices
    for (int count = 0; count < V - 1; count++) {
        // Pick the minimum distance vertex from the set of vertices not yet processed
        int u = minDistance(dist, sptSet, V);
       
        // Mark the picked vertex as processed
        sptSet[u] = true;
       
        printf("| Step %d | Selected vertex: %-2d                                        |\n", count + 1, u);
       
        printf("| Dist   | ");
        for (int i = 0; i < V; i++) {
            if (dist[i] == INF)
                printf("INF ");
            else
                printf("%-3d ", dist[i]);
        }
        printf("|\n");
       
        // Update dist value of the adjacent vertices of the picked vertex
        for (int v = 0; v < V; v++) {
            // Update dist[v] only if:
            // 1. There is an edge from u to v
            // 2. Vertex v is not in sptSet
            // 3. Distance to v through u is smaller than current value of dist[v]
            if (!sptSet[v] && graph[u][v] && dist[u] != INF
                && dist[u] + graph[u][v] < dist[v]) {
                parent[v] = u;
                dist[v] = dist[u] + graph[u][v];
                printf("|        | Updated vertex %-2d: new distance = %-3d                      |\n",
                       v, dist[v]);
            }
        }
        printf("+--------+------------------------------------------------------------+\n");
    }
   
    // Print the solution
    printSolution(dist, parent, src, V, "Dijkstra's");
}

// Optimized Dijkstra's algorithm using min heap
void dijkstraOptimized(int graph[MAX_VERTICES][MAX_VERTICES], int src, int V) {
    int dist[MAX_VERTICES];     // Distance from source to each vertex
    int parent[MAX_VERTICES];   // Parent array to store shortest path tree
   
    // Initialize min heap
    MinHeap* minHeap = createMinHeap(V);
   
    // Initialize all distances as INFINITE
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
        parent[i] = -1;
        minHeap->array[i] = (HeapNode){i, dist[i]};
        minHeap->position[i] = i;
    }
   
    // Make dist value of src vertex as 0
    dist[src] = 0;
    decreaseKey(minHeap, src, dist[src]);
   
    // Initially size of min heap is equal to V
    minHeap->size = V;
   
    printf("\nOptimized Dijkstra's Algorithm Steps (Using Min Heap):\n");
    printf("+--------+------------------------------------------------------------+\n");
    int step = 1;
   
    // Min heap contains all vertices that are not yet processed
    while (!isEmpty(minHeap)) {
        // Extract the vertex with minimum distance
        HeapNode extractedNode = extractMin(minHeap);
        int u = extractedNode.vertex;
       
        printf("| Step %d | Extracted vertex: %-2d with distance %-3d                      |\n",
               step++, u, dist[u]);
       
        // Traverse through all adjacent vertices of u
        for (int v = 0; v < V; v++) {
            // If there is an edge from u to v and v is not yet processed
            if (graph[u][v] && dist[u] != INF &&
                isInMinHeap(minHeap, v) &&
                dist[u] + graph[u][v] < dist[v]) {
               
                // Update distance and parent
                parent[v] = u;
                dist[v] = dist[u] + graph[u][v];
               
                // Update distance value in min heap
                decreaseKey(minHeap, v, dist[v]);
               
                printf("|        | Updated vertex %-2d: new distance = %-3d                      |\n",
                       v, dist[v]);
            }
        }
       
        printf("+--------+------------------------------------------------------------+\n");
    }
   
    // Free the heap
    free(minHeap->array);
    free(minHeap->position);
    free(minHeap);
   
    // Print the solution
    printSolution(dist, parent, src, V, "Optimized Dijkstra's");
}

// Function to implement Bellman-Ford algorithm
void bellmanFord(int graph[MAX_VERTICES][MAX_VERTICES], int src, int V) {
    int dist[MAX_VERTICES];    // Distance from source to each vertex
    int parent[MAX_VERTICES];  // Parent array to store shortest path tree
   
    // Initialize all distances as INFINITE and parent as -1
    for (int i = 0; i < V; i++) {
        dist[i] = INF;
        parent[i] = -1;
    }
    dist[src] = 0;
   
    // Convert the adjacency matrix to edge list for easier implementation
    Edge edges[MAX_VERTICES * MAX_VERTICES];
    int e = 0;
   
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (graph[i][j] != 0) {
                edges[e].src = i;
                edges[e].dest = j;
                edges[e].weight = graph[i][j];
                e++;
            }
        }
    }
   
    // Step by step results for Bellman-Ford algorithm
    printf("\nBellman-Ford Algorithm Steps:\n");
   
    // Relax all edges |V| - 1 times
    for (int i = 1; i <= V - 1; i++) {
        printf("+-------------+------------------------------------------------------+\n");
        printf("| Iteration %d |                                                      |\n", i);
        printf("+-------------+------------------------------------------------------+\n");
        bool no_changes = true;
       
        for (int j = 0; j < e; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int weight = edges[j].weight;
           
            if (dist[u] != INF && dist[u] + weight < dist[v]) {
                dist[v] = dist[u] + weight;
                parent[v] = u;
                printf("| Edge        | (%d -> %d) relaxed. New distance to %d = %-3d            |\n",
                       u, v, v, dist[v]);
                no_changes = false;
            }
        }
       
        printf("| Distances   | ");
        for (int k = 0; k < V; k++) {
            if (dist[k] == INF)
                printf("INF ");
            else
                printf("%-3d ", dist[k]);
        }
        printf("                             |\n");
        printf("+-------------+------------------------------------------------------+\n");
       
        // If no relaxation in this iteration, we can terminate early
        if (no_changes) {
            printf("| Note        | No changes in this iteration. Early termination.     |\n");
            printf("+-------------+------------------------------------------------------+\n");
            break;
        }
    }
   
    // Check for negative-weight cycles
    for (int i = 0; i < e; i++) {
        int u = edges[i].src;
        int v = edges[i].dest;
        int weight = edges[i].weight;
       
        if (dist[u] != INF && dist[u] + weight < dist[v]) {
            printf("\n| WARNING     | Graph contains negative weight cycle              |\n");
            printf("+------------+-----------------------------------------------------+\n");
            return;
        }
    }
   
    // Print the solution
    printSolution(dist, parent, src, V, "Bellman-Ford");
}

// Function to implement Floyd-Warshall algorithm
void floydWarshall(int graph[MAX_VERTICES][MAX_VERTICES], int V) {
    // Create distance and path matrices
    int dist[MAX_VERTICES][MAX_VERTICES];
    int path[MAX_VERTICES][MAX_VERTICES]; // For path reconstruction
   
    // Initialize distance and path matrices
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == j) {
                dist[i][j] = 0;
            } else if (graph[i][j] != 0) {
                dist[i][j] = graph[i][j];
            } else {
                dist[i][j] = INF;
            }
           
            // Initialize path matrix
            if (i == j || dist[i][j] == INF) {
                path[i][j] = -1;
            } else {
                path[i][j] = i;
            }
        }
    }
   
    printf("\nFloyd-Warshall Algorithm Steps:\n");
   
    // Compute all-pairs shortest paths
    for (int k = 0; k < V; k++) {
        printf("+-------------+------------------------------------------------------+\n");
        printf("| Iteration %d | Using vertex %d as intermediate                       |\n", k + 1, k);
        printf("+-------------+------------------------------------------------------+\n");
       
        // Print current distance matrix before updates
        printf("| Current    |\n");
        printf("| Distance   | ");
        for (int i = 0; i < V; i++) {
            printf("%-2d ", i);
        }
        printf("|\n| Matrix     | ");
        for (int i = 0; i < V; i++) {
            printf("---");
        }
        printf("|\n");
       
        for (int i = 0; i < V; i++) {
            printf("|           | ");
            for (int j = 0; j < V; j++) {
                if (dist[i][j] == INF)
                    printf("∞  ");
                else
                    printf("%-2d ", dist[i][j]);
            }
            printf("|\n");
        }
       
        // Update distance matrix
        bool updated = false;
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (dist[i][k] != INF && dist[k][j] != INF &&
                    dist[i][k] + dist[k][j] < dist[i][j]) {
                   
                    updated = true;
                    printf("| Update     | Path from %d to %d through %d: %d -> %d                |\n",
                           i, j, k, dist[i][j], dist[i][k] + dist[k][j]);
                   
                    dist[i][j] = dist[i][k] + dist[k][j];
                    path[i][j] = path[k][j];
                }
            }
        }
       
        if (!updated) {
            printf("| Note       | No paths were improved using vertex %d               |\n", k);
        }
       
        printf("+-------------+------------------------------------------------------+\n");
    }
   
    // Check for negative cycles (a vertex to itself with negative distance)
    for (int i = 0; i < V; i++) {
        if (dist[i][i] < 0) {
            printf("\n| WARNING    | Graph contains negative weight cycle              |\n");
            printf("+------------+-----------------------------------------------------+\n");
            return;
        }
    }
   
    // Print all-pairs shortest paths
    printAllPairsSolution(dist, path, V);
}

// Johnson's algorithm for all-pairs shortest paths
void johnsons(int graph[MAX_VERTICES][MAX_VERTICES], int V) {
    printf("\nJohnson's Algorithm Steps:\n");
    printf("+---------------+----------------------------------------------------+\n");
    printf("| Phase         | Description                                        |\n");
    printf("+---------------+----------------------------------------------------+\n");
    printf("| Preprocessing | Adding a new vertex and zero-weight edges          |\n");
   
    // Step 1: Add a new vertex with zero-weight edges to all other vertices
    int augmentedGraph[MAX_VERTICES + 1][MAX_VERTICES + 1] = {0};
    int newV = V + 1;
   
    // Copy original graph
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            augmentedGraph[i][j] = graph[i][j];
        }
    }
   
    // Add edges from new vertex to all others with weight 0
    for (int i = 0; i < V; i++) {
        augmentedGraph[V][i] = 0;
    }
   
    // Step 2: Run Bellman-Ford from the new vertex to compute h values
    int h[MAX_VERTICES + 1];    // Potential function values
    int parent[MAX_VERTICES + 1];
   
    // Initialize h values and parent array
    for (int i = 0; i <= V; i++) {
        h[i] = INF;
        parent[i] = -1;
    }
    h[V] = 0;
   
    // Convert the augmented adjacency matrix to edge list
    Edge edges[MAX_VERTICES * MAX_VERTICES + MAX_VERTICES];
    int e = 0;
   
    for (int i = 0; i <= V; i++) {
        for (int j = 0; j < V; j++) {
            if (i == V || augmentedGraph[i][j] != 0) {
                edges[e].src = i;
                edges[e].dest = j;
                edges[e].weight = (i == V) ? 0 : augmentedGraph[i][j];
                e++;
            }
        }
    }
   
    printf("| Bellman-Ford  | Computing potential function h(v) for each vertex   |\n");
   
    // Run Bellman-Ford
    for (int i = 1; i <= V; i++) {
        for (int j = 0; j < e; j++) {
            int u = edges[j].src;
            int v = edges[j].dest;
            int weight = edges[j].weight;
           
            if (h[u] != INF && h[u] + weight < h[v]) {
                h[v] = h[u] + weight;
                parent[v] = u;
            }
        }
    }
   
    // Check for negative-weight cycles
    for (int i = 0; i < e; i++) {
        int u = edges[i].src;
        int v = edges[i].dest;
        int weight = edges[i].weight;
       
        if (h[u] != INF && h[u] + weight < h[v]) {
            printf("| WARNING      | Graph contains negative weight cycle              |\n");
            printf("+---------------+----------------------------------------------------+\n");
            return;
        }
    }
   
    printf("| Reweighting  | Reweighting edges to eliminate negative weights      |\n");
   
    // Step 3: Reweight the edges
    int reweightedGraph[MAX_VERTICES][MAX_VERTICES] = {0};
   
    for (int i = 0; i < V; i++) {
        for (int j = 0; j < V; j++) {
            if (graph[i][j] != 0) {
                // Apply the reweighting formula: w'(u,v) = w(u,v) + h(u) - h(v)
                reweightedGraph[i][j] = graph[i][j] + h[i] - h[j];
                printf("| Edge (%d,%d)   | Original weight: %-3d  Reweighted: %-3d                |\n",
                       i, j, graph[i][j], reweightedGraph[i][j]);
            }
        }
    }
   
    printf("| Dijkstra     | Running Dijkstra from each vertex                    |\n");
   
    // Step 4: Run Dijkstra for each source vertex on the reweighted graph
    int allPairsDist[MAX_VERTICES][MAX_VERTICES];
   
    for (int i = 0; i < V; i++) {
        int dist[MAX_VERTICES];
        bool sptSet[MAX_VERTICES];
       
        // Initialize
        for (int j = 0; j < V; j++) {
            dist[j] = INF;
            sptSet[j] = false;
        }
        dist[i] = 0;
       
        // Run Dijkstra's algorithm
        for (int count = 0; count < V - 1; count++) {
            int u = minDistance(dist, sptSet, V);
            sptSet[u] = true;
           
            for (int v = 0; v < V; v++) {
                if (!sptSet[v] && reweightedGraph[u][v] &&
                    dist[u] != INF && dist[u] + reweightedGraph[u][v] < dist[v]) {
                    dist[v] = dist[u] + reweightedGraph[u][v];
                }
            }
        }
       
        // Step 5: Convert back to original weights
        for (int j = 0; j < V; j++) {
            if (dist[j] != INF) {
                // Apply the reverse reweighting: d(s,v) = d'(s,v) - h(s) + h(v)
                allPairsDist[i][j] = dist[j] - h[i] + h[j];
            } else {
                allPairsDist[i][j] = INF;
            }
        }
    }
   
    printf("| Conversion   | Converting distances back to original graph          |\n");
    printf("+---------------+----------------------------------------------------+\n");
   
    // Print result
    printf("\nJohnson's Algorithm Results (All-Pairs Shortest Paths):\n");
    printf("+-------+");
    for (int i = 0; i < V; i++) {
        printf("------+");
    }
    printf("\n| To→   |");
    for (int i = 0; i < V; i++) {
        printf(" %-4d |", i);
    }
    printf("\n| From↓ |");
    for (int i = 0; i < V; i++) {
        printf("------+");
    }
    printf("\n");
   
    for (int i = 0; i < V; i++) {
        printf("| %-5d |", i);
        for (int j = 0; j < V; j++) {
            if (allPairsDist[i][j] == INF)
                printf(" INF  |");
            else
                printf(" %-4d |", allPairsDist[i][j]);
        }
        printf("\n");
    }
   
    printf("+-------+");
    for (int i = 0; i < V; i++) {
        printf("------+");
    }
    printf("\n");
}

// Manhattan distance heuristic for A*
int manhattanDistance(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

// A* Search algorithm
void aStar(int graph[MAX_VERTICES][MAX_VERTICES], int src, int dest, int V) {
    // We need coordinates for nodes to calculate heuristic
    // For demonstration purposes, let's create some artificial coordinates
    int coords[MAX_VERTICES][2];
   
    // Generate grid-like coordinates
    int dimension = (int)sqrt(V);
    for (int i = 0; i < V; i++) {
        coords[i][0] = i % dimension;  // x coordinate
        coords[i][1] = i / dimension;  // y coordinate
    }
   
    // Initialize data structures
    AStarNode nodes[MAX_VERTICES];
    bool closedSet[MAX_VERTICES] = {false};
    int parent[MAX_VERTICES];
   
    // Initialize all nodes
    for (int i = 0; i < V; i++) {
        nodes[i].vertex = i;
        nodes[i].g_score = INF;
        nodes[i].f_score = INF;
        nodes[i].parent = -1;
        parent[i] = -1;
    }
   
    // Initialize source node
    nodes[src].g_score = 0;
    nodes[src].f_score = manhattanDistance(coords[src][0], coords[src][1],
                                           coords[dest][0], coords[dest][1]);
   
    printf("\nA* Search Algorithm Steps:\n");
    printf("+--------+------------------------------------------------------------+\n");
    printf("| Step   | Description                                                |\n");
    printf("+--------+------------------------------------------------------------+\n");
   
    int step = 1;
   
    // Main loop
    while (true) {
        // Find node with lowest f_score in the open set
        int current = -1;
        int minF = INF;
       
        for (int i = 0; i < V; i++) {
            if (!closedSet[i] && nodes[i].f_score < minF) {
                current = i;
                minF = nodes[i].f_score;
            }
        }
       
        // If we can't find any node in open set, path doesn't exist
        if (current == -1) {
            printf("| %6d | No path exists from %d to %d                              |\n",
                   step++, src, dest);
            printf("+--------+------------------------------------------------------------+\n");
            break;
        }
       
        // If we've reached the destination
        if (current == dest) {
            printf("| %6d | Reached destination %d. Search complete.                    |\n",
                   step++, dest);
            printf("+--------+------------------------------------------------------------+\n");
            break;
        }
       
        // Add current to closed set
        closedSet[current] = true;
       
        printf("| %6d | Exploring vertex: %-2d (g=%d, f=%d)                          |\n",
               step++, current, nodes[current].g_score, nodes[current].f_score);
       
        // Check all neighbors
        for (int neighbor = 0; neighbor < V; neighbor++) {
            // Skip if no edge or already in closed set
            if (graph[current][neighbor] == 0 || closedSet[neighbor])
                continue;
           
            // Calculate tentative g score
            int tentative_g = nodes[current].g_score + graph[current][neighbor];
           
            // If this path is better than any previous one
            if (tentative_g < nodes[neighbor].g_score) {
                // Update neighbor information
                parent[neighbor] = current;
                nodes[neighbor].parent = current;
                nodes[neighbor].g_score = tentative_g;
                nodes[neighbor].f_score = tentative_g +
                    manhattanDistance(coords[neighbor][0], coords[neighbor][1],
                                     coords[dest][0], coords[dest][1]);
               
                printf("|        | Updated vertex %-2d: g=%d, f=%d                              |\n",
                       neighbor, nodes[neighbor].g_score, nodes[neighbor].f_score);
            }
        }
    }
   
    // Reconstruct and print the path if destination was reached
    if (parent[dest] != -1) {
        int dist[MAX_VERTICES];
        for (int i = 0; i < V; i++) {
            dist[i] = (i == dest) ? nodes[dest].g_score : INF;
        }
       
        // Print the solution
        printf("\nA* Search Results:\n");
        printf("+---------+------------------------+--------------------------------+\n");
        printf("| Vertex  | Distance from Source   | Path                           |\n");
        printf("+---------+------------------------+--------------------------------+\n");
       
        printf("| %-7d | %-22d | ", dest, nodes[dest].g_score);
       
        // Reconstruct the path
        int path[MAX_VERTICES];
        int pathLen = 0;
        int current = dest;
       
        while (current != -1) {
            path[pathLen++] = current;
            current = parent[current];
        }
       
        // Print path in reverse order (from source to destination)
        for (int i = pathLen - 1; i >= 0; i--) {
            printf("%d", path[i]);
            if (i > 0) printf(" -> ");
        }
       
        printf(" |\n");
        printf("+---------+------------------------+--------------------------------+\n");
    }
}

// Function to create and analyze a sample graph
void createAndAnalyzeGraph() {
    int V, E;
    printf("Enter the number of vertices (max %d): ", MAX_VERTICES);
    scanf("%d", &V);
   
    // Initialize adjacency matrix
    int graph[MAX_VERTICES][MAX_VERTICES] = {0};
   
    printf("Enter the number of edges: ");
    scanf("%d", &E);
   
    printf("Enter the edges (source destination weight):\n");
    for (int i = 0; i < E; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
       
        if (u >= 0 && u < V && v >= 0 && v < V) {
            graph[u][v] = w;
        } else {
            printf("Invalid edge! Vertices must be between 0 and %d\n", V - 1);
            i--; // Retry this edge
        }
    }
   
    int source, destination;
    printf("Enter the source vertex: ");
    scanf("%d", &source);
   
    // Validate source
    if (source < 0 || source >= V) {
        printf("Invalid source vertex! Must be between 0 and %d\n", V - 1);
        return;
    }
   
    // Print graph information
    printf("\nGraph Information:\n");
    printf("Number of vertices: %d\n", V);
    printf("Number of edges: %d\n", E);
    printf("Adjacency Matrix:\n");
   
    printf("   ");
    for (int i = 0; i < V; i++) {
        printf("%2d ", i);
    }
    printf("\n");
   
    for (int i = 0; i < V; i++) {
        printf("%2d ", i);
        for (int j = 0; j < V; j++) {
            printf("%2d ", graph[i][j]);
        }
        printf("\n");
    }
   
    // Menu for algorithm selection
    int choice;
    do {
        printf("\nChoose algorithm to run:\n");
        printf("1. Dijkstra's Algorithm\n");
        printf("2. Optimized Dijkstra's Algorithm (Min Heap)\n");
        printf("3. Bellman-Ford Algorithm\n");
        printf("4. Floyd-Warshall Algorithm\n");
        printf("5. Johnson's Algorithm\n");
        printf("6. A* Search Algorithm\n");
        printf("0. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);
       
        switch (choice) {
            case 1:
                dijkstra(graph, source, V);
                break;
            case 2:
                dijkstraOptimized(graph, source, V);
                break;
            case 3:
                bellmanFord(graph, source, V);
                break;
            case 4:
                floydWarshall(graph, V);
                break;
            case 5:
                johnsons(graph, V);
                break;
            case 6:
                printf("Enter the destination vertex: ");
                scanf("%d", &destination);
                if (destination >= 0 && destination < V) {
                    aStar(graph, source, destination, V);
                } else {
                    printf("Invalid destination vertex! Must be between 0 and %d\n", V - 1);
                }
                break;
            case 0:
                printf("Exiting program.\n");
                break;
            default:
                printf("Invalid choice! Please try again.\n");
        }
    } while (choice != 0);
}

int main() {
    printf("===== SHORTEST PATH ALGORITHMS VISUALIZER =====\n\n");
    createAndAnalyzeGraph();
    return 0;
}

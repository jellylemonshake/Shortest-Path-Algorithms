#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

#define MAX_VERTICES 20
#define INF INT_MAX // Define infinity

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
    typedef struct {
        int src, dest, weight;
    } Edge;
    
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

int main() {
    int graph[MAX_VERTICES][MAX_VERTICES] = {0};
    int V, E, source;
    int u, v, w;
    
    // Get number of vertices
    printf("Enter the number of vertices (max %d): ", MAX_VERTICES);
    scanf("%d", &V);
    
    if (V <= 0 || V > MAX_VERTICES) {
        printf("Invalid number of vertices. Please enter a value between 1 and %d.\n", MAX_VERTICES);
        return 1;
    }
    
    // Get number of edges
    printf("Enter the number of edges: ");
    scanf("%d", &E);
    
    if (E < 0 || E > V * (V - 1)) {
        printf("Invalid number of edges. A directed graph with %d vertices can have at most %d edges.\n", 
               V, V * (V - 1));
        return 1;
    }
    
    // Get edge information
    printf("\nEnter edge information (source vertex, destination vertex, weight):\n");
    for (int i = 0; i < E; i++) {
        printf("Edge %d: ", i + 1);
        scanf("%d %d %d", &u, &v, &w);
        
        if (u < 0 || u >= V || v < 0 || v >= V) {
            printf("Invalid vertex. Vertices should be between 0 and %d.\n", V - 1);
            i--; // Retry this edge
            continue;
        }
        
        graph[u][v] = w;
    }
    
    // Get source vertex
    printf("\nEnter the source vertex (0 to %d): ", V - 1);
    scanf("%d", &source);
    
    if (source < 0 || source >= V) {
        printf("Invalid source vertex. Please choose a vertex between 0 and %d.\n", V - 1);
        return 1;
    }
    
    // Print the graph
    printf("\nGraph Representation (Adjacency Matrix):\n");
    printf("+");
    for (int i = 0; i < V; i++) {
        printf("---+");
    }
    printf("\n");
    
    for (int i = 0; i < V; i++) {
        printf("| ");
        for (int j = 0; j < V; j++) {
            printf("%2d| ", graph[i][j]);
        }
        printf("\n+");
        for (int j = 0; j < V; j++) {
            printf("---+");
        }
        printf("\n");
    }
    
    printf("\nFinding shortest paths from source vertex %d\n", source);
    
    // Ask which algorithm to run
    int choice;
    printf("\nChoose algorithm to run:\n");
    printf("1. Bellman-Ford Algorithm\n");
    printf("2. Dijkstra's Algorithm\n");
    printf("3. Both algorithms\n");
    printf("Enter your choice (1-3): ");
    scanf("%d", &choice);
    
    switch (choice) {
        case 1:
            bellmanFord(graph, source, V);
            break;
        case 2:
            dijkstra(graph, source, V);
            break;
        case 3:
            bellmanFord(graph, source, V);
            dijkstra(graph, source, V);
            break;
        default:
            printf("Invalid choice. Running both algorithms by default.\n");
            bellmanFord(graph, source, V);
            dijkstra(graph, source, V);
    }
    
  
    return 0;
}

import numpy as np
import time
def calculate_cut_weight(adjacency_matrix : np.array, x: set, y:set):
    total_cut_weight = 0
    for _x in x:
        total_cut_weight += np.sum(adjacency_matrix[_x, list(y)])
    return total_cut_weight

def update_sigma(adjacency_matrix: np.array, sigma_x : np.array, sigma_y : np.array, from_x_to_y: bool, best_vertex:int):
    num_vertices = adjacency_matrix.shape[0]
    for u in range(num_vertices):
        if u != best_vertex:
            if from_x_to_y:
                sigma_x[u] -= adjacency_matrix[u, best_vertex]
                sigma_y[u] += adjacency_matrix[u, best_vertex]
            else:
                sigma_x[u] += adjacency_matrix[u, best_vertex]
                sigma_y[u] -= adjacency_matrix[u, best_vertex]
    return sigma_x,sigma_y

def local_search(adjacency_matrix: np.array, x: set, y: set):
    num_iterations = 0
    num_vertices = adjacency_matrix.shape[0]
    x_indices = np.array(list(x))
    y_indices = np.array(list(y))
    sigma_x = adjacency_matrix[:, x_indices].sum(axis=1) if len(x_indices) > 0 else np.zeros(num_vertices)
    sigma_y = adjacency_matrix[:, y_indices].sum(axis=1) if len(y_indices) > 0 else np.zeros(num_vertices)
    improvement = True
    while improvement:
        num_iterations += 1
        improvement = False
        best_improvement = 0
        best_vertex = None
        from_x_to_y = None
        for v in range(num_vertices):
            if v in x:
                del_v = sigma_x[v] - sigma_y[v]
                if del_v > best_improvement:
                    best_improvement = del_v
                    best_vertex = v
                    from_x_to_y = True
            else:
                del_v = sigma_y[v] - sigma_x[v]
                if del_v > best_improvement:
                    best_improvement = del_v
                    best_vertex = v
                    from_x_to_y = False

        if best_improvement > 0 and best_vertex is not None:
            improvement = True
            if from_x_to_y:
                x.remove(best_vertex)
                y.add(best_vertex)
            else:
                y.remove(best_vertex)
                x.add(best_vertex)
            
            sigma_x, sigma_y = update_sigma(adjacency_matrix=adjacency_matrix, 
                                            sigma_x=sigma_x,
                                            sigma_y=sigma_y,
                                            from_x_to_y=from_x_to_y,
                                            best_vertex=best_vertex)
    total_weight = calculate_cut_weight(adjacency_matrix, x, y)
    return total_weight, num_iterations


def greedy_heuristic_for_max_cut(adjacency_matrix: np.array):
    num_vertices = adjacency_matrix.shape[0]
    vertices = [i for i in range(num_vertices)]
    x = set()
    y = set()
    arg_x,arg_y = np.unravel_index(adjacency_matrix.argmax(), adjacency_matrix.shape)
    x.add(arg_x)
    y.add(arg_y)
    vertices.remove(arg_x)
    vertices.remove(arg_y)
    for z in vertices:
        weight_x = np.sum(adjacency_matrix[list(y), z])
        weight_y = np.sum(adjacency_matrix[list(x), z])
        if (weight_x > weight_y):
            x.add(z)
        else:
            y.add(z)
    return calculate_cut_weight(adjacency_matrix=adjacency_matrix, x=x, y=y)
    
def semi_greedy_heuristic_for_max_cut(adjacency_matrix: np.array, alpha, mode = 'value'):
    num_vertices = adjacency_matrix.shape[0]
    #print(num_vertices)
    vertices = [i for i in range(num_vertices)]
    sigma_x = np.zeros(num_vertices)
    sigma_y = np.zeros(num_vertices)
    greedy_value = np.zeros(num_vertices)
    x = set()
    y = set()
    arg_x,arg_y = np.unravel_index(adjacency_matrix.argmax(), adjacency_matrix.shape)
    #print(arg_x, arg_y)
    x.add(arg_x)
    y.add(arg_y)
    vertices.remove(arg_x)
    vertices.remove(arg_y)

    for z in vertices:
        sigma_x[z] = adjacency_matrix[z,arg_x]
        sigma_y[z] = adjacency_matrix[z,arg_y]

    while vertices:
        for z in vertices:
            greedy_value[z] = max(sigma_x[z], sigma_y[z])
        if (mode == 'value'):
            w_min = min(np.min(sigma_x[list(vertices)]), np.min(sigma_y[list(vertices)]))
            w_max = max(np.max(sigma_x[list(vertices)]), np.max(sigma_y[list(vertices)]))

            #print(w_max, w_min)
            mu = w_min + alpha*(w_max - w_min)
            #print(f'mu:{mu}')
            RCL = [ v for v in vertices if greedy_value[v] >= mu]
        elif (mode == 'cardinality'):
            indices_sorted = np.argsort(greedy_value)[::-1]
            RCL = [v for v in indices_sorted if v in vertices][:5]
        #print(f"RCL Size : {len(RCL)}")
        if RCL:
            rand = np.random.choice(RCL) 
            if sigma_x[rand] > sigma_y[rand]:
                y.add(rand)
                for z in vertices:
                    sigma_y[z] += adjacency_matrix[z, rand]
            else:
                x.add(rand)
                for z in vertices:
                    sigma_x[z] += adjacency_matrix[z, rand]
            vertices.remove(rand)
    return x,y



def randomized_maxcut(adjacency_matrix : np.array):
    num_vertices = adjacency_matrix.shape[0]
    total_cut_weight = 0
    for _ in range(num_vertices):
        x = set()
        y = set()
        #print(f"x : {x}")
        #print(f"y : {y}")
        for v in range(num_vertices):
            choice = np.random.random()  
            if (choice >= 0.5):
                x.add(v)
            else:
                y.add(v)
        total_cut_weight += calculate_cut_weight(adjacency_matrix=adjacency_matrix,x=x,y=y)    
    return total_cut_weight // num_vertices      


def grasp(adjacency_matrix: np.array, max_iterations: int):
    best_weight = 0
    for i in range(max_iterations):
       # print(i)
        X,Y = semi_greedy_heuristic_for_max_cut(adjacency_matrix=adjacency_matrix, alpha=0.3)
        weight, _ = local_search(adjacency_matrix=adjacency_matrix, x=X, y=Y)
        if i == 0 or weight > best_weight:  
            best_weight = weight 
            
    return best_weight

        

if (__name__ == '__main__'):
    vert_edg = input()
    vert_edg = vert_edg.split(' ')
    num_vertices = int(vert_edg[0])
    num_edges = int(vert_edg[1])
    adjacency_matrix = np.zeros((num_vertices,num_vertices))
    for _ in range(num_edges):
        v1_v2_w = input()
        v1_v2_w = v1_v2_w.split(' ')
        v1 = int(v1_v2_w[0])
        v2 = int(v1_v2_w[1])
        w = int(v1_v2_w[2])
        adjacency_matrix[v1-1,v2-1] = w
        adjacency_matrix[v2-1,v1-1] = w
    time_begin = time.time()
    randomized_max_cut = randomized_maxcut(adjacency_matrix=adjacency_matrix)
    print(f"Randomized Max-Cut - {randomized_max_cut}")

    greedy_max_cut = greedy_heuristic_for_max_cut(adjacency_matrix=adjacency_matrix)
    print(f"Greedy Max-Cut - {greedy_max_cut}")
    
    x_semi_greedy, y_semi_greedy = semi_greedy_heuristic_for_max_cut(adjacency_matrix=adjacency_matrix, alpha=0.3)
    semi_greedy_max_cut = calculate_cut_weight(adjacency_matrix=adjacency_matrix, x=x_semi_greedy, y=y_semi_greedy)
    print(f"Semi Greedy Max-Cut - {semi_greedy_max_cut}")

    grasp_max_cut = grasp(adjacency_matrix=adjacency_matrix,max_iterations=100)
    print(f"GRASP Max-Cut - {grasp_max_cut}")

    time_end = time.time()

    print(f"Runtime : {time_end - time_begin}")


"""
3 3
1 2 1
1 3 2
2 3 1`

"""


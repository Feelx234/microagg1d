import numpy as np
from numba import njit
from numba.typed import List
from numba.types import int16, int32, int64



@njit([(int16[:], int32), (int32[:], int32), (int64[:], int64)],cache=True)
def my_bincount(arr, min_lenght):
    """The same as numpy bincount, but works on unsigned integers as well"""
    if min_lenght <= 0:
        m=arr.max() # pragma: no cover
    else:
        m = min_lenght
    out = np.zeros(m+1, dtype=arr.dtype)
    for i in range(len(arr)):
        out[arr[i]]+=np.int32(1)
    return out


@njit([(int32[:,:], int32), (int64[:,:], int64)], cache=True)
def get_neighbors_from_edges(edges, num_nodes):
    """ transforms the edges into two arrays the first arrays indicates ranges into the second array
        the second array contains the in neighbors of those nodes indicated in array 1
    input :
        takes in edges in the format a -> b (first column a, second column b)
        assumes the nodes are consecutively labeled from 0 to num_nodes
        first column are the sending nodes while the second column are the receiving nodes

    output:
    starting_positions[i] ..starting_positions[i+1] contains all in neighbors of node i
    in_neighbors
    """
    if num_nodes ==0:
        num_nodes = edges.ravel().max()
    else:
        assert num_nodes > 0
        num_nodes -= 1
    degrees = my_bincount(edges.ravel(), min_lenght=np.int32(num_nodes))

    # starting_positions[i] ..starting_positions[i+1] contains all in neighbors of node i
    starting_positions = np.empty(degrees.shape[0]+1, dtype=np.int32)
    starting_positions[0]=0
    starting_positions[1:] = degrees.cumsum()
    current_index = starting_positions.copy()

    neighbors = np.zeros(2*edges.shape[0], dtype=np.int32)

    for i in range(edges.shape[0]):
        l = edges[i,0]
        r = edges[i,1]
        neighbors[current_index[r]] = l
        neighbors[current_index[l]] = r
        current_index[r] += 1
        current_index[l] += 1

    return starting_positions, neighbors, degrees


@njit(cache=True)
def filter_edges(edges, keep_nodes):
    out_edges = np.empty_like(edges)
    n = 0
    for i in range(edges.shape[0]):
        l = edges[i,0]
        r = edges[i,1]
        if l in keep_nodes and r in keep_nodes:
            out_edges[n,0]=l
            out_edges[n,1]=r
            n+=1
    return out_edges[:n,:]


@njit(cache=True)
def compute_dists(arr):
    n, d = arr.shape
    dist = np.full((n,n),np.inf,dtype=np.float64)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sum(np.square(arr[i,:]-arr[j,:]))
            dist[i,j]=d
            dist[j,i]=d
    return dist

def get_k_closest(dist, k):
    partition = np.argpartition(dist,k, axis=1)
    return partition[:,:k]


def forest(closest_neighbors):
    n = closest_neighbors.shape[0]
    k = closest_neighbors.shape[1]

    edges = np.empty((n,2), dtype=np.int64)
    current_tree = np.empty(n, dtype=np.int64)
    current_size = 0

    partition = np.full(n, -1, dtype=np.int64)
    components = List()
    num_edges = 0
    
    order = np.random.permutation(n)
    for v in order:
        if partition[v]>=0:
            continue
        current_tree[0] = v
        current_size = 1
        partition[v] = len(components)
        other_partition = -1
        while current_size < k and other_partition==-1:
            for neigh in closest_neighbors[v]: # this may be randomized as well
                # print(v, neigh, partition[v], partition[neigh] )
                if partition[neigh]==len(components):
                    continue
                edges[num_edges,0] = v
                edges[num_edges,1] = neigh
                num_edges+=1
                if partition[neigh]>=0:
                    other_partition = partition[neigh]
                    break
                current_tree[current_size]=neigh
                current_size+=1
                v = neigh
                partition[v]=len(components)
        if other_partition == -1:
            components.append(current_tree[0:current_size].copy())
        else:
            for node in current_tree[0:current_size]:
                partition[node] = other_partition
    return partition, components, edges[:num_edges,:]




@njit
def forest2(closest_neighbors):
    n = closest_neighbors.shape[0]
    k = closest_neighbors.shape[1]

    edges = np.empty((n,2), dtype=np.int64)

    partitions = list()
    for node in range(n):
        l = list()
        l.append(node)
        partitions.append(l)
    small_partitions = np.arange(n)
    num_small_partitions = n
    position = np.arange(n)
    out_degree = np.zeros(n, dtype=np.int16)
    partition = np.arange(n, dtype=np.int64)


    num_edges = 0

    while num_small_partitions > 0:
        index = np.random.randint(num_small_partitions)
        p = small_partitions[index]
        #print("P", p)
        p_len = len(partitions[p])
        # print(num_small_partitions)
        if p_len >= k or p_len==0:
            small_partitions[index] = small_partitions[num_small_partitions-1]
            num_small_partitions-=1
            continue
        #print("B", p_len)
        order = np.random.permutation(p_len)
        found=False
        for o in order:
            v = partitions[p][o]
            # print(v, out_degree[v])
            if out_degree[v]>0:
                continue
            other_partition = -1
            for neigh in closest_neighbors[v]: # this may be randomized as well
                # print(v, neigh, partition[v], partition[neigh] )
                if partition[neigh]==partition[v]:
                    continue
                out_degree[v] = 1
                edges[num_edges,0] = v
                edges[num_edges,1] = neigh
                num_edges+=1
                p1 = partition[v]
                p2 = partition[neigh]
                if len(partitions[p1]) > len(partitions[p2]):
                    p1 = partition[neigh]
                    p2 = partition[v]
                # now p1 is the smaller partition
                for node in partitions[p1]: # rename node in partition p1
                    partition[node] = p2
                partitions[p2].extend(partitions[p1])
                partitions[p1].clear()
                found = True
                # print("found")
                break
            if found:
                break
        assert found, found
                

    components = list()
    for p in partitions:
        if len(p)>0:
            arr = np.empty(len(p), dtype=np.int64)
            for i, x in enumerate(p):
                arr[i]=x
            
            components.append(arr)
    return components, edges[:num_edges,:]



def find_subtree(pivot, avoid,  start_pos, neighbors, sub_tree_size, component):
    out = np.empty(len(start_pos), dtype=np.int64)
    out[0]=pivot
    out_len = 1
    q = np.empty(len(start_pos), dtype=np.int64)
    q[0] = pivot
    q_len = 1
    # print(sub_tree_size)
    sub_tree_size[avoid]=1
    sub_tree_size[pivot]=1
    
    while q_len>0:
        q_len-=1
        node = q[q_len]
        start = start_pos[node]
        stop = start_pos[node+1]
        
        for neigh in neighbors[start:stop]:
            # print("N", node, neigh)
            if sub_tree_size[neigh]==1:
                continue
            else:
                out[out_len]=neigh
                out_len+=1
                q[q_len]=neigh
                q_len+=1
                sub_tree_size[neigh]=1
    for node in component:
        sub_tree_size[node]=0
    return out[:out_len]


def decompose_components(components, edges, k, num_nodes):
    start_pos, neighbors, messages_still_pending = get_neighbors_from_edges(edges, num_nodes)
    # print(messages_still_pending)
    sub_tree_size=np.zeros_like(messages_still_pending)
    s = list()
    s.append(edges[0,0])
    s.pop()

    candidate_components = list()
    for component in components:
        steiners = s.copy()
        candidate_components.append((steiners, component))

    final_components = list()
    while len(candidate_components)>0:
        # print("main")
        steiners, component = candidate_components.pop()
        comp_edges = filter_edges(edges, component)
        # print(component)
        start_pos, neighbors, messages_still_pending = get_neighbors_from_edges(comp_edges, num_nodes)
        res = split_component(steiners, component, k, start_pos, neighbors, messages_still_pending, sub_tree_size)
        for steiners, comp in res:
            # print("out", comp)
            if len(comp)-len(steiners) > max(2*k-1, 3*k-5):
                candidate_components.append((steiners, comp))
            else:
                final_components.append((steiners, comp))
        
    return final_components





def split_component(steiners, component, k, start_pos, neighbors, messages_still_pending, sub_tree_size):
    component = np.random.permutation(component)
    # print(component)
    s = len(component) - len(steiners)
    l = list()

    if s <= max(2*k-1, 3*k-5):
        l.append((steiners, component))
        return l
    steiners1 = steiners.copy()
    steiners2 = steiners.copy()
    for i, pivot in enumerate(component):
        degree_pivot = start_pos[pivot+1]-start_pos[pivot]
        if pivot in steiners:
            continue
        if degree_pivot==1:
            continue
        sub_tree_sizes = hang_tree(pivot, component, start_pos, neighbors, messages_still_pending, sub_tree_size, steiners)
        j = np.argmax(sub_tree_sizes[:,1])
        v = sub_tree_sizes[j,0]
        phi = sub_tree_sizes[j,1]
        #print("phi", pivot, phi)
        # print(sub_tree_sizes)
        if s-phi < k-1:
            continue
        if phi>=k and s-phi >=k:
            comp1 = find_subtree(pivot, v, start_pos, neighbors, sub_tree_size, component)
            comp2 = find_subtree(v, pivot, start_pos, neighbors, sub_tree_size, component)
            l.append((steiners1, comp1))
            l.append((steiners2, comp2))
            # print("A", pivot, list(l))
            return l
        elif s-phi == k-1:
            if v in steiners:
                continue
            comp1_p = find_subtree(pivot, v, start_pos, neighbors, sub_tree_size, component)
            # print("B1", comp1_p)
            comp1 = np.empty(len(comp1_p)+1, dtype=comp1_p.dtype)
            comp1[0]=v
            comp1[1:]=comp1_p[:]
            comp2 = find_subtree(v, pivot, start_pos, neighbors, sub_tree_size, component)
            # print("B", v, pivot, comp1, comp2)
            steiners2.append(v)
            l.append((steiners1, comp1))
            l.append((steiners2, comp2))
            #print("B", pivot, v, list(l))
            return l
        elif s-phi == k-1:
            comp1 = find_subtree(pivot, v, start_pos, neighbors, sub_tree_size, component)
            comp2 = find_subtree(v, pivot, start_pos, neighbors, sub_tree_size, component)
            comp2_p = np.empty(len(comp2)+1, dtype=comp1.dtype)
            comp2[0]=pivot
            comp2[1:]=comp2_p
            steiners2.append(v)
            l.append((steiners1, comp1))
            l.append((steiners2, comp2))
            # print("C", pivot, list(l))
            return l
        else:
            # print("D", pivot, list(l))
            num_subtrees = sub_tree_sizes.shape[0]
            order = np.random.permutation(num_subtrees)
            i_order = 0
            tree_size = 0
            while tree_size < k-1:
                tree_size+=sub_tree_sizes[order[i_order],1]
                i_order+=1
            #print(sub_tree_sizes)
            #print(tree_size)
            add_pivot_to_1 = True
            if s-tree_size == k-1:
                add_pivot_to_1=False
            else:
                add_pivot_to_1=True
            comp1 = np.empty(tree_size+1, dtype=np.int64)
            comp2 = np.empty(s-tree_size+1, dtype=np.int64)
            comp1[0]=pivot
            comp2[0]=pivot
            i_comp1 = 1
            i_comp2 = 1
            if add_pivot_to_1:
                steiners2.append(pivot) # the pivot is assigned to comp1, thus 2 is steiner
            else:
                steiners1.append(pivot)
            
            for j in range(num_subtrees):
                sub_node = sub_tree_sizes[order[j],0]
                comp_tmp = find_subtree(sub_node, pivot, start_pos, neighbors, sub_tree_size, component)
                #print(j, sub_node, comp_tmp)
                for sub_tree_member in comp_tmp:
                    if j < i_order:
                        comp1[i_comp1] = sub_tree_member
                        i_comp1+=1
                    else:
                        comp2[i_comp2] = sub_tree_member
                        i_comp2+=1
            l.append((steiners1, comp1))
            l.append((steiners2, comp2))
            return l
        # l1 = split_component(steiners1, comp1, k, start_pos, neighbors, messages_still_pending, sub_tree_size, component)
        # l.extend(l1)    
        # l2 = split_component(steiners2, comp2, k, start_pos, neighbors, messages_still_pending, sub_tree_size, component)
        # l.extend(l2)
        
        return l
    assert False  


def hang_tree(pivot, nodes, start_pos, neighbors, messages_still_pending, sub_tree_size, steiners):
    # print()
    degree_pivot = start_pos[pivot+1]-start_pos[pivot]
    out = np.empty((degree_pivot, 2), dtype=np.int64)
    num_out=0
    leaves = np.empty_like(nodes)
    num_leaves=0

    # Fill initial leaves
    for node in nodes:
        if node not in steiners:
            sub_tree_size[node] = 1
        else:
            sub_tree_size[node] = 0
        degree_node = start_pos[node+1]-start_pos[node]
        if degree_node!=1:
            continue
        leaves[num_leaves]=node
        num_leaves+=1
        messages_still_pending[node]=0 # leaf nodes need no messages
    # print("pivot", pivot)
    # print(messages_still_pending)
    while num_leaves > 0:
        # pop node from the queue
        num_leaves -=1
        node = leaves[num_leaves]
        # print("node", node)
        if node == pivot: # pivot never needs to send messages
            continue
        # propagate to neighbors
        start = start_pos[node]
        stop = start_pos[node+1]
        # print(node, start, stop, neighbors[start:stop], len(neighbors))
        for neigh in neighbors[start:stop]:
            if messages_still_pending[neigh]<=1:
                continue
            # print("  ", node, neigh)
            sub_tree_size[neigh]+=sub_tree_size[node]
            
            if neigh == pivot:
                out[num_out,0]=node
                out[num_out,1]=sub_tree_size[node]
                num_out+=1
                #print("root", node)
                break
            if messages_still_pending[neigh]>2:
                messages_still_pending[neigh]-=1
            elif messages_still_pending[neigh]==2: # this node is now a leaf so enqueu
                messages_still_pending[neigh]-=1
                # print(node, neigh, sub_tree_size[node], sub_tree_size[neigh])
                
                leaves[num_leaves] = neigh
                num_leaves+=1
                # print("new_leaf", neigh)
            break
    #print(messages_still_pending, pivot)
    # reset global arrays
    for node in nodes:
        degree = start_pos[node+1]-start_pos[node]
        messages_still_pending[node]=degree
        sub_tree_size[node]=0
    assert num_out == degree_pivot, (pivot, num_out, degree_pivot)
    return out       
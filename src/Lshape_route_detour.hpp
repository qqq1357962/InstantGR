#include "graph.hpp"
#include <omp.h>
#include <atomic>
#include <unistd.h>
#include <queue>
#define INF_LAYER 20
#define MAX_LAYER 10
#define MIN_ROUTE_LAYER 1
#define MAX_DEPTH 5000
// node_cnt*81
__managed__ double *cost_edges;
__managed__ int *best_change;
__managed__ int edge_cnt;

namespace Lshape_route_detour {

//declaration
void Lshape_route_detour(vector<int> &nets2route);
__managed__ int *macroBorder;
int *macroBorder_cpu;
int cntCongested = 0;
int totalEdgeNum = 0;

__managed__ int *node_cnt_sum, *nodes, *par_nodes, *from, *layer_range;
int node_cnt_estimate;
int parent_cnt_estimate;
__managed__ int *child_num; 
__managed__ int *child_num_sum; 
__managed__ int *in_degree;
__managed__ int *currentChildIDX;

__managed__ int *par_num;
__managed__ int *par_num_sum;
__managed__ int *locks;
__managed__ double *childCosts;
__managed__ int *childCosts_road;
__managed__ unsigned long long *packed_information;

__managed__ unsigned long long *best_path;

__managed__ int *layer_output;
__managed__ double *costs;
__managed__ int *fixed_layers;
__managed__ int *node_net_idx;
__managed__ int *node_net_idx2;

__managed__ int *lock_gpu;

__managed__ int *node_depth;
__managed__ int *net_depth;

__managed__ int *batch_depth;
__managed__ int *depth_node;
__managed__ int *depth_node_cnt;
__managed__ int log;
bool *congestionView_cpu;
float *congestionView_xsum_cpu;
float *congestionView_ysum_cpu;
int *node_cnt_sum_cpu, *node_depth_cpu, *net_depth_cpu, *batch_depth_cnt_cpu, *depth_node_cnt_cpu, *depth_node_cpu,
    *nodes_cpu, *node_net_idx_cpu, *node_net_idx2_cpu, *child_num_cpu, *child_num_sum_cpu, *par_num_cpu, *par_num_sum_cpu,
    *par_nodes_cpu, *currentChildIDX_cpu, *depthID2nodeSequence;

__device__ void atomicMinDouble(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) {
            break;
        }
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

__global__ void init_min_child_costs(int limit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    childCosts[index] = INF;
}

#define COST_BITS 32
#define COST_SHIFT (64 - COST_BITS)

__device__ int32_t float_to_ordered_int32(float f)
{
    int32_t i = __float_as_int(f);
    return (i >= 0) ? i : i ^ 0x7FFFFFFF;
}

__device__ float ordered_int32_to_float(int32_t i)
{
    return __int_as_float((i >= 0) ? i : i ^ 0x7FFFFFFF);
}

__device__ unsigned long long packCostAndNodeId(float cost, int node_id) {
    int32_t cost_int = float_to_ordered_int32(cost);
    unsigned long long packed = ((unsigned long long)cost_int << COST_SHIFT) | (unsigned int)node_id;
    return packed;
}

__device__ void unpackCostAndNodeId(unsigned long long packed, double* cost, int* node_id) {
    int32_t cost_int = (int32_t)(packed >> COST_SHIFT);
    *cost = (double)ordered_int32_to_float(cost_int);
    *node_id = (int)(packed & ((1ULL << COST_SHIFT) - 1));
}

__device__ const unsigned long long init_cost_with_road = 7208259049189261824;

__global__ void init_road(int limit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    childCosts_road[index] = 200000000;
    packed_information[index] = init_cost_with_road;
}

__global__ void init_costs(int limit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>limit)
    {
        return;
    }
    costs[index] = INF;
}

__global__ void precompute_cost_edge(int par_sum)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int edge_id = index / 9;
    if(edge_id>=par_sum) return;
    int target_node_idx = par_nodes[edge_id];
    int from_layer = index%9;
    int from_node_idx = currentChildIDX[edge_id] / 10;
    int tox = nodes[target_node_idx] / Y % X, toy = nodes[target_node_idx] % Y;
    int fromx = nodes[from_node_idx] / Y % X, fromy = nodes[from_node_idx] % Y;
    if(((from_layer & 1 ^ DIR) == 0 && fromy != toy)||((from_layer & 1 ^ DIR) == 1 && fromx != tox))
    {
        for(int to_layer = 0; to_layer < 9; to_layer++)
        cost_edges[edge_id * 81 + from_layer * 9 + to_layer] = INF;
        return;
    }
    assert(fromx == tox || fromy == toy);
    int distance = abs(tox - fromx) + abs(toy - fromy);
    if(distance==0) return;
    int dd=distance / min(distance, 20);
    bool isVertical = (fromx == tox);
    int start_search_pos = isVertical?min(fromy, toy):min(fromx, tox);
    int end_search_pos = isVertical?max(fromy, toy):max(fromx, tox);
    for(int to_layer = 0; to_layer < 9; to_layer++)
    {
        if((to_layer%2)!=(from_layer%2)) continue;
        double min_costt = INF;
        int decide_pos = -1;
        assert(start_search_pos >= 0);
        assert(end_search_pos >= 0);
        assert(dd>0);
        for(int search_pos = start_search_pos+dd; ; search_pos+=dd)
        {
            if(search_pos >= end_search_pos)
            {
                break;
            }
            int search_x, search_y;
            if(isVertical)
            {
                search_x = fromx;
                search_y = search_pos;
            }
            else{
                search_x = search_pos;
                search_y = fromy;
            }
            assert(search_x>=0);
            assert(search_y>=0);
            double via_cost_tmp = 0;
            for(int ll = min(from_layer, to_layer); ll < max(from_layer, to_layer); ll++)
            {
                via_cost_tmp += vcost[IDX(ll, search_x, search_y)];
            }
            double cost_tmp = via_cost_tmp;
            if(isVertical)
            {
                cost_tmp += (graph::wire_segment_cost(from_layer, fromx, fromx, min(fromy, search_y), max(fromy, search_y))
                            + graph::wire_segment_cost(to_layer, tox, tox, min(toy, search_y), max(toy, search_y)));
            }else{
                cost_tmp += (graph::wire_segment_cost(from_layer, min(fromx, search_x), max(fromx, search_x), fromy, fromy)
                            + graph::wire_segment_cost(to_layer, min(tox, search_x), max(tox, search_x), toy, toy));
            }
            if(cost_tmp < min_costt)
            {
                min_costt = cost_tmp;
                decide_pos = search_pos;
            }
        }
        cost_edges[edge_id * 81 + from_layer * 9 + to_layer] = min_costt;
        best_change[edge_id * 81 + from_layer * 9 + to_layer] = decide_pos;
    }
}

__global__ void Lshape_route_node_cuda(int shift, int end_shift) {
    int node_sequence = blockIdx.x * blockDim.x + threadIdx.x + shift;
    if(node_sequence>=end_shift)
    {
        return;
    }
    int node_idx = depth_node[node_sequence];
    int parent_num_cur = par_num_sum[node_idx + 1]-par_num_sum[node_idx];
    int fixed_layer_low = 1 + nodes[node_idx] / X / Y;
    int x = nodes[node_idx] / Y % X, y = nodes[node_idx] % Y;
    int fixed_layer_high = fixed_layer_low==10?0:fixed_layer_low;
    int cur_child_num = child_num_sum[node_idx + 1]-child_num_sum[node_idx];

    unsigned long long *cur_best_path = best_path + child_num_sum[node_idx] * MAX_LAYER;
    unsigned long long *cur_packed_information = packed_information + child_num_sum[node_idx] * MAX_LAYER;
    double minChildCosts[6];
    int bestPaths[6];
    for (int lowLayerIndex = MIN_ROUTE_LAYER; lowLayerIndex <= fixed_layer_low; lowLayerIndex++) {
        for(int cid = 0; cid < cur_child_num; cid++)
        {
            minChildCosts[cid] = INF;
        }
        double via_cost = 0;
        for (int layerIndex = lowLayerIndex; layerIndex < (L + 1); layerIndex++) {
            if(layerIndex>lowLayerIndex)
            {   
                via_cost += vcost[IDX(layerIndex - 2, x, y)];
            }
            for (int childIndex = 0; childIndex < cur_child_num; childIndex++) {
                double cost;
                int nodeddd;
                unpackCostAndNodeId(cur_packed_information[childIndex * MAX_LAYER + layerIndex], &cost, &nodeddd);
                assert(nodeddd>=0);
                double cur_child_cost = cost;
                if (cur_child_cost < minChildCosts[childIndex]) {
                    minChildCosts[childIndex] = cur_child_cost;
                    bestPaths[childIndex] = nodeddd * MAX_LAYER + layerIndex;
                }
            }
            if (layerIndex >= fixed_layer_high) {
                double cost = via_cost;
                for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                {
                    cost += minChildCosts[childIndex];
                }
                if (cost<INF && cost < costs[node_idx*MAX_LAYER+layerIndex]) {
                    costs[node_idx*MAX_LAYER+layerIndex] = cost;
                    for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                    {
                        cur_best_path[childIndex * MAX_LAYER + layerIndex] = bestPaths[childIndex];
                    }
                }
            }
        }
        for (int layerIndex = (L + 1) - 2; layerIndex >= lowLayerIndex; layerIndex--) {//
            if (costs[node_idx*MAX_LAYER+layerIndex + 1] < costs[node_idx*MAX_LAYER+layerIndex]) {
                costs[node_idx*MAX_LAYER+layerIndex] = costs[node_idx*MAX_LAYER+layerIndex + 1];
                for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                {
                    cur_best_path[childIndex * MAX_LAYER + layerIndex] = cur_best_path[childIndex * MAX_LAYER + layerIndex + 1];
                }
            }
        }
    }
    int node_x = nodes[node_idx] / Y % X, node_y = nodes[node_idx] % Y; 

    for(int par_id = 0; par_id < parent_num_cur; par_id++)
    {
        int parent_IDX = par_nodes[par_num_sum[node_idx] + par_id];
        int child_index_of_current_node = currentChildIDX[par_num_sum[node_idx] + par_id]%10;
        unsigned long long *parent_packed_information = packed_information + child_num_sum[parent_IDX] * MAX_LAYER;
        int px = nodes[parent_IDX] / Y % X, py = nodes[parent_IDX] % Y; 
        assert(px == node_x || py == node_y);
        for(int layer = MIN_ROUTE_LAYER; layer<MAX_LAYER; layer++)
        {
            if((layer & 1 ^ DIR) == 1 && node_y != py) continue;
            if((layer & 1 ^ DIR) == 0 && node_x != px) continue;
            int index_ = child_index_of_current_node * MAX_LAYER + layer;
            double cost = costs[node_idx * MAX_LAYER + layer] + graph::wire_segment_cost(layer-1, min(node_x, px), max(node_x, px), min(node_y, py), max(node_y, py));
            atomicMin(parent_packed_information + index_, packCostAndNodeId(cost, node_idx));
        }
    }
}

__global__ void Lshape_route_node_layer_change_cuda(int shift, int end_shift) {
    int node_sequence = blockIdx.x * blockDim.x + threadIdx.x + shift;
    if(node_sequence>=end_shift)
    {
        return;
    }
    int node_idx = depth_node[node_sequence];
    int parent_num_cur = par_num_sum[node_idx + 1]-par_num_sum[node_idx];
    int fixed_layer_low = 1 + nodes[node_idx] / X / Y;
    int x = nodes[node_idx] / Y % X, y = nodes[node_idx] % Y;
    int fixed_layer_high = fixed_layer_low==10?0:fixed_layer_low;
    int cur_child_num = child_num_sum[node_idx + 1]-child_num_sum[node_idx];

    unsigned long long *cur_best_path = best_path + child_num_sum[node_idx] * MAX_LAYER;
    unsigned long long *cur_packed_information = packed_information + child_num_sum[node_idx] * MAX_LAYER;
    double minChildCosts[5];
    unsigned long long bestPaths[5];
    for (int lowLayerIndex = MIN_ROUTE_LAYER; lowLayerIndex <= fixed_layer_low; lowLayerIndex++) {
        for(int cid = 0; cid<cur_child_num; cid++)
        {
            minChildCosts[cid] = INF;
        }
        double via_cost = 0;
        for (int layerIndex = lowLayerIndex; layerIndex < (L + 1); layerIndex++) {
            if(layerIndex>lowLayerIndex)
            {   
                via_cost += vcost[IDX(layerIndex - 2, x, y)];
            }
            for (int childIndex = 0; childIndex < cur_child_num; childIndex++) {
                double cost;
                int nodeddd;
                unpackCostAndNodeId(cur_packed_information[childIndex * MAX_LAYER + layerIndex], &cost, &nodeddd);
                double cur_child_cost = cost;
                assert(nodeddd >= 0);
                if (cur_child_cost < minChildCosts[childIndex]) {
                    minChildCosts[childIndex] = cur_child_cost;
                    bestPaths[childIndex] = (unsigned long long)nodeddd * MAX_LAYER + layerIndex;
                }
            }
            if (layerIndex >= fixed_layer_high) {
                double cost = via_cost;
                for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                {
                    cost += minChildCosts[childIndex];
                }
                if (cost<INF && cost < costs[node_idx * MAX_LAYER+layerIndex]) {
                    costs[node_idx * MAX_LAYER + layerIndex] = cost;
                    for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                    {
                        cur_best_path[childIndex * MAX_LAYER + layerIndex] = bestPaths[childIndex];
                    }
                }
            }
        }
        for (int layerIndex = (L + 1) - 2; layerIndex >= lowLayerIndex; layerIndex--) {//
            if (costs[node_idx * MAX_LAYER + layerIndex + 1] < costs[node_idx * MAX_LAYER + layerIndex]) {
                costs[node_idx * MAX_LAYER + layerIndex] = costs[node_idx * MAX_LAYER + layerIndex + 1];
                for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                {
                    cur_best_path[childIndex * MAX_LAYER + layerIndex] = cur_best_path[childIndex * MAX_LAYER + layerIndex + 1];
                }
            }
        }
    }
    int node_x = nodes[node_idx] / Y % X, node_y = nodes[node_idx] % Y; 
    unsigned long long packed_information_cache[11];
    for(int par_id = 0; par_id < parent_num_cur; par_id++)
    {
        int parent_IDX = par_nodes[par_num_sum[node_idx] + par_id];
        int child_index_of_current_node = currentChildIDX[par_num_sum[node_idx] + par_id] % 10;
        int parent_shift_base = child_num_sum[parent_IDX] * MAX_LAYER;
        unsigned long long *parent_packed_information = packed_information + parent_shift_base;
        int px = nodes[parent_IDX] / Y % X, py = nodes[parent_IDX] % Y;
        int minx = min(node_x, px), maxx = max(node_x, px), miny = min(node_y, py), maxy = max(node_y, py);
        assert(px == node_x || py == node_y);
        
        for(int target_layer=0; target_layer<MAX_LAYER; target_layer++)
        {
            if((target_layer<MIN_ROUTE_LAYER)||(target_layer>=MAX_LAYER)) continue;
            int index_ = child_index_of_current_node * MAX_LAYER + target_layer;
            packed_information_cache[target_layer] = *(parent_packed_information + index_);
        }
        for(int layer = MIN_ROUTE_LAYER; layer < MAX_LAYER; layer++) // output=1-9
        {
            if((layer & 1 ^ DIR) == 1 && node_y != py) continue;
            if((layer & 1 ^ DIR) == 0 && node_x != px) continue;
            double cost_base = costs[node_idx * MAX_LAYER + layer];
            assert(cost_base<INF);
            for(int target_layer = MIN_ROUTE_LAYER + (1 - layer%2); target_layer< MAX_LAYER; target_layer+=2)
            {
                if((target_layer < MIN_ROUTE_LAYER) || (target_layer >= MAX_LAYER)) continue;
                double cost2 = INF;
                int possibility_choosed = 82;
                if((target_layer == layer))
                {
                    cost2 = cost_base + graph::wire_segment_cost(layer-1, minx, maxx, miny, maxy);
                }
                else if(abs(node_x - px) + abs(node_y - py)>4){
                    int possibility_sequence = (layer-1) * 9 + (target_layer-1);
                    double c_e = cost_edges[(par_num_sum[node_idx] + par_id) * 81 + possibility_sequence];
                    if(c_e < INF){
                        cost2 = cost_base + c_e;
                        possibility_choosed = possibility_sequence;
                    } else{
                        continue;
                    }
                }
                else{
                    continue;
                }
                volatile unsigned long long new_value = packCostAndNodeId(cost2, node_idx * 100 + possibility_choosed);
                if(new_value<packed_information_cache[target_layer])
                {
                    packed_information_cache[target_layer] = new_value;
                }
            }
        }
        
        for(int target_layer = MIN_ROUTE_LAYER; target_layer< MAX_LAYER; target_layer++)
        {
            if((target_layer<MIN_ROUTE_LAYER)||(target_layer>=MAX_LAYER)) continue;
            int index_ = child_index_of_current_node * MAX_LAYER + target_layer;
            atomicMin(parent_packed_information + index_, packed_information_cache[target_layer]);
        }   
    }
}

__global__ void get_routing_tree_cuda(int shift, int end_shift, int depth, int stamp) {
    int node_sequence = blockIdx.x * blockDim.x + threadIdx.x + shift;
    if(node_sequence>=end_shift)
    {
        return;
    }
    int node_id = depth_node[node_sequence];
    int net_id = node_net_idx[node_id];
    assert(net_id<NET_NUM);
    assert(pin_acc_num_phase2[net_id]>=0);
    int *net_routes = routes_phase2 + pin_acc_num_phase2[net_id] * ROUTE_PER_PIN_PHASE2;
    unsigned long long *cur_best_path = best_path + child_num_sum[node_id] * MAX_LAYER;

    int l = nodes[node_id] / Y / X;
    int cur_x = nodes[node_id] / Y % X, cur_y = nodes[node_id] % Y;
    if(par_num_sum[node_id + 1]-par_num_sum[node_id]==0)
    {
        int min_layer = 0;
        double min_cost = costs[node_id * MAX_LAYER];
        for(int layer = 1; layer < MAX_LAYER; layer++)
        {
            if(costs[node_id * MAX_LAYER + layer] < min_cost)
            {
                min_cost = costs[ node_id * MAX_LAYER + layer];
                min_layer = layer;
            }
        }
        layer_output[node_id] = min_layer;
        net_routes[0] = 1;
    } else{
        int par_layer = -1;
        int par_idx = -1;
        int par_sequence = -1;
        for(int par_id = 0; par_id< par_num_sum[node_id + 1]-par_num_sum[node_id]; par_id++)
        {
            int par_node = par_nodes[par_num_sum[node_id]+par_id];
            if(layer_output[par_node]>=0)
            {
                par_idx = par_node;
                par_sequence = par_id;
                par_layer = layer_output[par_node];
                int child_index_of_current_node = currentChildIDX[par_num_sum[node_id]+par_sequence]%10;
                unsigned long long *par_best_path = best_path + child_num_sum[par_idx] * MAX_LAYER;
                unsigned long long path = par_best_path[child_index_of_current_node * MAX_LAYER + par_layer];
                int child_idx = path / MAX_LAYER;
                if(child_idx == node_id)
                {
                    layer_output[node_id] = path % MAX_LAYER;
                    int px = nodes[par_idx] / Y % X, py = nodes[par_idx] % Y;
                    assert(px == cur_x || py == cur_y); 
                    if(px == cur_x && cur_y != py)
                    {
                        graph::atomic_add_unit_demand_wire_segment(layer_output[node_id] - 1, px, px, min(py,cur_y), max(py,cur_y), stamp);
                        int idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(layer_output[node_id] - 1, px, min(py,cur_y));
                        net_routes[idd1 + 1] = IDX(layer_output[node_id] - 1, px, max(py,cur_y));
                        assert(idd1 + 1<ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1]-pin_acc_num[net_id]));
                    }
                    else if(py == cur_y && cur_x != px)
                    {
                        graph::atomic_add_unit_demand_wire_segment(layer_output[node_id] - 1, min(px,cur_x), max(px,cur_x), py, py, stamp);
                        int idd1 = atomicAdd(net_routes,2);
                        net_routes[idd1] = IDX(layer_output[node_id] - 1, min(px,cur_x), py);
                        net_routes[idd1 + 1] = IDX(layer_output[node_id] - 1, max(px,cur_x), py);
                        assert(idd1 + 1 < ROUTE_PER_PIN_PHASE2 * (pin_acc_num[net_id + 1] - pin_acc_num[net_id]));
                    }
                    break;
                }else{
                    layer_output[node_id] = -1;
                }
            }
        }
        if(par_layer==-1)
        {
            layer_output[node_id] = -1;
            return;
        }
        int child_index_of_current_node = currentChildIDX[par_num_sum[node_id]+par_sequence]%10;
        unsigned long long *par_best_path = best_path + child_num_sum[par_idx] * MAX_LAYER;
        unsigned long long path = par_best_path[child_index_of_current_node * MAX_LAYER + par_layer];
        unsigned long long child_idx = path / MAX_LAYER;
        if( child_idx != node_id)
        {
            layer_output[node_id] = -1;
            return;
        }
    }
    int num_child = child_num_sum[node_id + 1] - child_num_sum[node_id];
    int minl = l + 1;
    int maxl = (l + 1)==MAX_LAYER?1:minl;
    minl = min(minl,layer_output[node_id]);
    maxl = max(maxl,layer_output[node_id]);
    if(num_child>0)
    {
        for(int child_id=0; child_id < num_child; child_id++)
        {
            int layer_of_child = cur_best_path[child_id * MAX_LAYER + layer_output[node_id]] % MAX_LAYER;
            assert(layer_of_child!=0);
            minl = min(layer_of_child, minl);
            maxl = max(layer_of_child, maxl);
        }
    }
    if(minl<maxl)
    {
        int idd1 = atomicAdd(net_routes,2);
        net_routes[idd1] = IDX(minl - 1, cur_x, cur_y);
        net_routes[idd1 + 1] = IDX(maxl - 1, cur_x, cur_y);
        assert(idd1 + 1 < ROUTE_PER_PIN_PHASE2 * (pin_acc_num[net_id + 1]-pin_acc_num[net_id]));
    }
}


__global__ void get_routing_tree_layer_change_cuda(int shift, int end_shift, int depth, int stamp) {
    int node_sequence = blockIdx.x * blockDim.x + threadIdx.x + shift;
    if(node_sequence>=end_shift)
    {
        return;
    }
    int node_id = depth_node[node_sequence];
    int net_id = node_net_idx[node_id];
    assert(net_id<NET_NUM);
    assert(pin_acc_num_phase2[net_id]>=0);
    int *net_routes = routes_phase2 + pin_acc_num_phase2[net_id] * ROUTE_PER_PIN_PHASE2;

    unsigned long long *cur_best_path = best_path + child_num_sum[node_id] * MAX_LAYER;

    int l = nodes[node_id] / Y / X;
    int cur_x = nodes[node_id] / Y % X, cur_y = nodes[node_id] % Y;
    if(par_num_sum[node_id + 1]-par_num_sum[node_id]==0)
    {
        int min_layer = 0;
        double min_cost = costs[node_id * MAX_LAYER];
        for(int layer = 1; layer < MAX_LAYER; layer++)
        {
            if(costs[node_id * MAX_LAYER + layer] < min_cost)
            {
                min_cost = costs[ node_id * MAX_LAYER + layer];
                min_layer = layer;
            }
        }
        layer_output[node_id] = min_layer;
        net_routes[0] = 1;
    } else{
        int par_layer = -1;
        int par_idx = -1;
        int par_sequence = -1;
        for(int par_id = 0; par_id< par_num_sum[node_id + 1]-par_num_sum[node_id]; par_id++)
        {
            int edge_id = par_num_sum[node_id]+par_id;
            int par_node = par_nodes[par_num_sum[node_id]+par_id];
            if(layer_output[par_node]>0)
            {
                par_idx = par_node;
                par_sequence = par_id;
                par_layer = layer_output[par_node];
                int child_index_of_current_node = currentChildIDX[par_num_sum[node_id]+par_sequence]%10;
                unsigned long long *par_best_path = best_path + child_num_sum[par_idx] * MAX_LAYER;
                unsigned long long path = par_best_path[child_index_of_current_node * MAX_LAYER + par_layer];
                unsigned long long child_idx = path / MAX_LAYER / 100;
                unsigned long long possibility_choosed = (path / MAX_LAYER) % 100;
                if(child_idx == node_id)
                {
                    if(possibility_choosed==82)
                    {
                        layer_output[node_id] = path % MAX_LAYER;
                    }
                    else{
                        layer_output[node_id] = possibility_choosed / 9 + 1;
                    }
                    int px = nodes[par_idx] / Y % X, py = nodes[par_idx] % Y;
                    assert(px==cur_x||py==cur_y); 
                    if(px==cur_x && cur_y!=py)
                    {
                        if(possibility_choosed==82)
                        {
                            graph::atomic_add_unit_demand_wire_segment(layer_output[node_id] - 1, px, px, min(py,cur_y), max(py,cur_y), stamp);
                            int idd1 = atomicAdd(net_routes,2);
                            assert(idd1 + 1 < ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1]-pin_acc_num[net_id]));
                            net_routes[idd1] = IDX(layer_output[node_id] - 1, px, min(py,cur_y));
                            net_routes[idd1 + 1] = IDX(layer_output[node_id] - 1, px, max(py,cur_y));
                            assert(IDX(layer_output[node_id] - 1, px, min(py,cur_y))>=0);
                            assert(idd1 + 1<ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1]-pin_acc_num[net_id]));
                        }
                        else{
                            assert(possibility_choosed<81);
                            int from_layer = possibility_choosed / 9;
                            int to_layer = possibility_choosed % 9;
                            assert((path % MAX_LAYER)==(to_layer + 1));
                            int change_pos = best_change[edge_id * 81 + from_layer * 9 + to_layer];

                            assert(change_pos>min(cur_y, py));
                            assert(change_pos<max(cur_y, py));
                            assert((from_layer%2)==(to_layer%2));
                            assert((from_layer%2)==0);// vertical
                            graph::atomic_add_unit_demand_wire_segment(from_layer, cur_x, cur_x, min(cur_y,change_pos), max(cur_y,change_pos), stamp);
                            graph::atomic_add_unit_demand_wire_segment(to_layer, cur_x, cur_x, min(py,change_pos), max(py,change_pos), stamp);
                            int idd1 = atomicAdd(net_routes,6);
                            net_routes[idd1] = IDX(from_layer, cur_x, min(cur_y,change_pos));
                            net_routes[idd1 + 1] = IDX(from_layer, cur_x,  max(cur_y,change_pos));
                            assert(from_layer!=to_layer);
                            net_routes[idd1 + 2] = IDX(min(from_layer, to_layer), cur_x, change_pos);
                            net_routes[idd1 + 3] = IDX(max(from_layer, to_layer), cur_x, change_pos);
                            net_routes[idd1 + 4] = IDX(to_layer, cur_x, min(py,change_pos));
                            net_routes[idd1 + 5] = IDX(to_layer, cur_x,  max(py,change_pos));
                            assert(idd1 + 5<ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1] - pin_acc_num[net_id]));
                        }
                    }
                    else if(py==cur_y && cur_x != px)
                    {
                        if(possibility_choosed==82)
                        {
                            graph::atomic_add_unit_demand_wire_segment(layer_output[node_id] - 1, min(px,cur_x), max(px,cur_x), py, py, stamp);
                            int idd1 = atomicAdd(net_routes,2);
                            net_routes[idd1] = IDX(layer_output[node_id] - 1, min(px, cur_x), py);
                            net_routes[idd1 + 1] = IDX(layer_output[node_id] - 1, max(px, cur_x), py);
                            assert(IDX(layer_output[node_id] - 1, min(px,cur_x), py) >= 0);
                            assert(idd1 + 1 < ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1]-pin_acc_num[net_id]));
                        }
                        else{
                            assert(possibility_choosed<81);
                            int from_layer = possibility_choosed / 9;
                            int to_layer = possibility_choosed % 9;
                            assert((path % MAX_LAYER)==(to_layer + 1));
                            int change_pos = best_change[edge_id * 81 + from_layer * 9 + to_layer];
                            assert(change_pos > min(cur_x, px));
                            assert(change_pos < max(cur_x, px));
                            assert((from_layer % 2)==(to_layer % 2));
                            assert((from_layer % 2)==1);
                            graph::atomic_add_unit_demand_wire_segment(from_layer, min(cur_x, change_pos), max(cur_x, change_pos), cur_y, cur_y, stamp);
                            graph::atomic_add_unit_demand_wire_segment(to_layer, min(px, change_pos), max(px, change_pos), cur_y, cur_y, stamp);
                            int idd1 = atomicAdd(net_routes, 6);
                            net_routes[idd1] = IDX(from_layer, min(cur_x, change_pos), cur_y);
                            net_routes[idd1 + 1] = IDX(from_layer, max(cur_x, change_pos), cur_y);
                            assert(from_layer!=to_layer);
                            net_routes[idd1 + 2] = IDX(min(from_layer, to_layer), change_pos, cur_y);
                            net_routes[idd1 + 3] = IDX(max(from_layer, to_layer), change_pos, cur_y);
                            net_routes[idd1 + 4] = IDX(to_layer, min(px,change_pos), cur_y);
                            net_routes[idd1 + 5] = IDX(to_layer, max(px,change_pos), cur_y);
                            assert(idd1 + 5 < ROUTE_PER_PIN_PHASE2 * (pin_acc_num[net_id + 1] - pin_acc_num[net_id]));
                        }
                    }
                    break;
                }else{
                    layer_output[node_id] = -1;
                }
            }
        }
        if(par_layer==-1)
        {
            layer_output[node_id] = -1;
            return;
        }
        int child_index_of_current_node = currentChildIDX[par_num_sum[node_id] + par_sequence]%10;
        unsigned long long *par_best_path = best_path + child_num_sum[par_idx] * MAX_LAYER;
        unsigned long long path = par_best_path[child_index_of_current_node * MAX_LAYER + par_layer];
        unsigned long long child_idx = path / MAX_LAYER / 100;
        if( child_idx != node_id)
        {
            layer_output[node_id] = -1;
            return;
        }
    }
    int num_child = child_num_sum[node_id + 1] - child_num_sum[node_id];
    int minl = l + 1; //l:0-8-->2nd - 10th layers
    int maxl = (l + 1)==MAX_LAYER?1:minl;
    minl = min(minl,layer_output[node_id]);
    maxl = max(maxl,layer_output[node_id]);
    assert(num_child>=0);
    if(num_child>0)
    {
        for(int child_id=0; child_id<num_child; child_id++)
        {
            int layer_of_child = cur_best_path[child_id * MAX_LAYER + layer_output[node_id]] % MAX_LAYER;
            assert(layer_of_child!=0);
            minl = min(layer_of_child, minl);
            maxl = max(layer_of_child, maxl);
        }
    }
    if(minl<maxl)
    {
        int idd1 = atomicAdd(net_routes,2);
        net_routes[idd1] = IDX(minl - 1, cur_x, cur_y);
        net_routes[idd1 + 1] = IDX(maxl - 1, cur_x, cur_y);
        assert(IDX(minl - 1, cur_x, cur_y)>=0);
        assert(idd1 + 1<ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1]-pin_acc_num[net_id]));
    }
}

__global__ void initWithLargeValue(int* data, int value, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        data[index] = value;
    }
}
void process_net(int thread_idx, vector<int> &nets2route, int thread_num, std::atomic<int>& currentNetId) {
    int is_show = false;
    while (true) {
        int netId = currentNetId.fetch_add(1);
        if (netId >= nets2route.size()) {
            break;
        }
        nets[nets2route[netId]].generate_detours_reconstruct(congestionView_cpu, congestionView_xsum_cpu, congestionView_ysum_cpu, false, is_show);
    }
}

void multithreaded_processing(vector<int> &nets2route) {
    std::vector<std::thread> threads;
    int max_threads = 8;
    threads.reserve(max_threads);
    std::atomic<int> currentNetId(0);
    for (int i = 0; i < max_threads; ++i) {
        threads.emplace_back(process_net, i, std::ref(nets2route), max_threads, std::ref(currentNetId));
    }
    for (auto& t : threads) {
        t.join();
    }
}

void Lshape_route_detour_wrap(vector<int> &nets2route) {
    double DAG_start_time = elapsed_time();
    if(nets2route.size()==0)
    {
        return;
    }
    sort(nets2route.begin(), nets2route.end(), [] (int l, int r) {
        return nets[l].hpwl > nets[r].hpwl;
    });
    congestionView_cpu = new bool[X * Y * sizeof(bool)];
    congestionView_xsum_cpu = new float[X * Y * sizeof(float)];
    congestionView_ysum_cpu = new float[X * Y * sizeof(float)];
    cudaMemcpy(congestionView_cpu, congestion, X * Y * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(congestionView_xsum_cpu, congestion_xsum, X * Y * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(congestionView_ysum_cpu, congestion_ysum, X * Y * sizeof(float), cudaMemcpyDeviceToHost);
    if(LOG) printf("[%5.1f] Generating h,v segments Starts\n", elapsed_time());
    for(int i = 0; i < nets2route.size(); i++) {
        if(nets[nets2route[i]].rsmt.size()<1)
        {
            nets[nets2route[i]].construct_rsmt();
        }
    }
    multithreaded_processing(nets2route);
    if(LOG) printf("[%5.1f] Generating h,v segments Ends\n", elapsed_time());
    if(LOG) printf("[%5.1f] Generating batches Starts\n", elapsed_time());
    auto batches = generate_batches_rsmt_gpu(nets2route, 300000);
    if(LOG) printf("[%5.1f] Generating batches Ends\n", elapsed_time());
    int net_cnt_estimate=0;
    int node_num_max=0;
    int par_num_max=0;
    print_GPU_memory_usage();
    for(int ii=0; ii<batches.size(); ii++)
    {
        int tmp = 0;
        int tmp2 = 0;
        if(batches[ii].size() > net_cnt_estimate)
        {
            net_cnt_estimate = batches[ii].size();
        }
        for(int j=0; j<batches[ii].size(); j++)
        {
            auto &graph_x = nets[batches[ii][j]].rsmt;
            tmp+=nets[batches[ii][j]].node_index_cnt;
            tmp2+=nets[batches[ii][j]].par_num_sum_cpu[nets[batches[ii][j]].node_index_cnt];
        }
        if(tmp>node_num_max)
        {
            node_num_max =tmp;
        }
        if(tmp2>par_num_max)
        {
            par_num_max =tmp2;
        }
    }

    net_cnt_estimate += 5;
    node_cnt_estimate = node_num_max + 10;
    parent_cnt_estimate = par_num_max + 10;
    if(LOG) printf("[%5.1f] Lshape_route Starts\n", elapsed_time());
    cudaMalloc(&node_cnt_sum, net_cnt_estimate * sizeof(int));
    cudaMalloc(&nodes, node_cnt_estimate * sizeof(int));
    cudaMalloc(&net_depth, net_cnt_estimate * sizeof(int));
    cudaMalloc(&batch_depth, (batches.size() + 1) * sizeof(int));
    cudaMalloc(&child_num_sum, node_cnt_estimate * sizeof(int));
    cudaMalloc(&par_num_sum, node_cnt_estimate * sizeof(int));
    cudaMalloc(&node_net_idx, node_cnt_estimate * sizeof(int));
    cudaMalloc(&node_net_idx2, node_cnt_estimate * sizeof(int));
    cudaMalloc(&node_depth, node_cnt_estimate * sizeof(int));
    cudaMalloc(&depth_node, node_cnt_estimate * sizeof(int));
    cudaMalloc(&layer_range, node_cnt_estimate * sizeof(int));
    cudaMalloc(&costs, node_cnt_estimate* MAX_LAYER * sizeof(double));
    cudaMalloc(&locks, parent_cnt_estimate* MAX_LAYER * sizeof(int));
    cudaMemset(locks, 0, sizeof(int) * parent_cnt_estimate* MAX_LAYER);
    cudaMalloc(&layer_output, node_cnt_estimate * sizeof(int));
    cudaMalloc(&par_nodes, parent_cnt_estimate * sizeof(int));
    cudaMalloc(&cost_edges, parent_cnt_estimate * 81 * sizeof(double));
    cudaMalloc(&best_change, parent_cnt_estimate * 81 * sizeof(int));
    cudaMalloc(&best_path, parent_cnt_estimate*MAX_LAYER * sizeof(unsigned long long));
    cudaMalloc(&childCosts, parent_cnt_estimate*MAX_LAYER * sizeof(double));
    cudaMalloc(&childCosts_road, parent_cnt_estimate*MAX_LAYER * sizeof(int));
    cudaMalloc(&packed_information, parent_cnt_estimate*MAX_LAYER * sizeof(unsigned long long));
    cudaMalloc(&currentChildIDX, parent_cnt_estimate * sizeof(int));
    ///////////////////////////////////  cpu arrays init starts  ////////////////////////////////////////
    node_cnt_sum_cpu = new int[net_cnt_estimate]();
    int reserve_node_num = node_cnt_estimate;
    int biggest_depth = MAX_DEPTH;
    node_depth_cpu = new int[reserve_node_num]();
    net_depth_cpu = new int[net_cnt_estimate]();
    batch_depth_cnt_cpu = new int[batches.size() + 1]();
    depth_node_cnt_cpu = new int[biggest_depth*(batches.size() + 1)]();
    depth_node_cpu = new int[reserve_node_num]();
    nodes_cpu = new int[reserve_node_num]();
    node_net_idx_cpu = new int[reserve_node_num]();
    node_net_idx2_cpu = new int[reserve_node_num]();
    child_num_cpu = new int[reserve_node_num]();
    child_num_sum_cpu = new int[reserve_node_num]();
    par_num_cpu = new int[reserve_node_num]();
    par_num_sum_cpu = new int[reserve_node_num]();
    par_nodes_cpu = new int[parent_cnt_estimate]();
    currentChildIDX_cpu = new int[parent_cnt_estimate]();
    depthID2nodeSequence = new int[batches.size()*MAX_DEPTH];
    ///////////////////////////////////  cpu arrays init ends  ////////////////////////////////////////
    // for(int ii = 0; ii < batches.size(); ii++)
    for(int ii = batches.size()-1; ii >=0; ii--)
    {
        Lshape_route_detour(batches[ii]);
    }
    

    cudaFree(node_cnt_sum);
    cudaFree(par_nodes);
    cudaFree(nodes);
    cudaFree(from);
    cudaFree(layer_range);
    cudaFree(in_degree);
    cudaFree(currentChildIDX);
    cudaFree(par_num_sum);
    cudaFree(locks);
    cudaFree(layer_output);
    cudaFree(fixed_layers);
    cudaFree(node_net_idx);
    cudaFree(node_net_idx2);
    cudaFree(lock_gpu);
    cudaFree(node_depth);
    cudaFree(net_depth);
    cudaFree(batch_depth);
    cudaFree(depth_node);
    cudaFree(depth_node_cnt);
    cudaFree(childCosts);
    cudaFree(childCosts_road);
    cudaFree(packed_information);
    cudaFree(best_path);
    delete[] congestionView_cpu;
    delete[] node_cnt_sum_cpu;
    delete[] node_depth_cpu;
    delete[] net_depth_cpu;
    delete[] batch_depth_cnt_cpu;
    delete[] depth_node_cnt_cpu;
    delete[] depth_node_cpu;
    delete[] nodes_cpu;
    delete[] node_net_idx_cpu;
    delete[] node_net_idx2_cpu;
    delete[] child_num_cpu;
    delete[] child_num_sum_cpu;
    delete[] par_num_cpu;
    delete[] par_num_sum_cpu;
    delete[] par_nodes_cpu;
    delete[] currentChildIDX_cpu;
    if(LOG) printf("[%5.1f] Lshape_route Ends\n", elapsed_time());
    DAG_time = elapsed_time() - DAG_start_time;
}


void Lshape_route_detour(vector<int> &nets2route) {
    vector<vector<int>> batches;
    batches.push_back(nets2route);
    vector<int> batch_cnt_sum(batches.size() + 1, 0);
    for(int i = 0; i < batches.size(); i++) {
        batch_cnt_sum[i + 1] = batch_cnt_sum[i] + batches[i].size();
        for(int j = 0; j < batches[i].size(); j++)
        {
            int net_id = batch_cnt_sum[i] + j;
            nets2route[net_id] = batches[i][j];
        }
    }
    int net_cnt = nets2route.size();
    int node_cnt = 0;
    int par_cnt = 0;
    for(auto net_id : nets2route)
    {
        node_cnt += nets[net_id].node_index_cnt;
        par_cnt += nets[net_id].par_num_sum_cpu[nets[net_id].node_index_cnt];
    }
    memset(node_cnt_sum_cpu, 0, (net_cnt + 1) * sizeof(int));
    int batch_reserve_node = node_cnt + 10;
    int reserve_node_num = min(node_cnt_estimate, batch_reserve_node);//to be optimized
    int reserve_par_num = min(parent_cnt_estimate, par_cnt+10);
    memset(net_depth_cpu, 0, net_cnt * sizeof(int));
    memset(batch_depth_cnt_cpu, 0, (batches.size() + 1) * sizeof(int));
    memset(depth_node_cnt_cpu, 0, MAX_DEPTH*(batches.size() + 1) * sizeof(int));
    memset(child_num_sum_cpu, 0, reserve_node_num * sizeof(int));
    memset(par_num_cpu, 0, reserve_node_num * sizeof(int));
    memset(par_num_sum_cpu, 0, reserve_node_num * sizeof(int));
    memset(depthID2nodeSequence, 0, batches.size()*MAX_DEPTH * sizeof(int));
    ////////////////////////////////////////////// cpu array memset ends //////////////////////////////////////////////////
    for(int b_id=0; b_id<batches.size(); b_id++)
    for(int j=0; j< batches[b_id].size(); j++){
        int net_idx = batches[b_id][j];
        auto &graph_x = nets[net_idx].rsmt;
        int select_root = nets[net_idx].select_root;
        int id = batch_cnt_sum[b_id] + j;
        int net_num_nodes = nets[net_idx].node_index_cnt;
        node_cnt_sum_cpu[id + 1] = net_num_nodes;
        for(int n_i= 0; n_i < net_num_nodes; n_i++)
        { 
            int node_id = node_cnt_sum_cpu[id] + n_i;
            nodes_cpu[node_id] = nets[net_idx].nodes_cpu[n_i];
            child_num_cpu[node_id] = nets[net_idx].child_num_cpu[n_i];
            child_num_sum_cpu[node_id + 1] = child_num_sum_cpu[node_id] + child_num_cpu[node_id];
            node_depth_cpu[node_id] = nets[net_idx].node_depth_cpu[n_i];
            int depth = node_depth_cpu[node_id];
            batch_depth_cnt_cpu[b_id + 1] = max(batch_depth_cnt_cpu[b_id + 1], node_depth_cpu[node_id] + 1);
            net_depth_cpu[id] = max(net_depth_cpu[id], depth);
            par_num_cpu[node_id] = nets[net_idx].par_num_cpu[n_i];
            par_num_sum_cpu[node_id + 1] = par_num_sum_cpu[node_id] + par_num_cpu[node_id];
            node_net_idx_cpu[node_id] = net_idx;
            node_net_idx2_cpu[node_id] = id;
        }
        int par_num_total = nets[net_idx].par_num_sum_cpu[net_num_nodes];
        int node_start = node_cnt_sum_cpu[id];
        int pid_total = 0;
        for(int n_i= 0; n_i < net_num_nodes; n_i++)
        {
            int node_id = node_cnt_sum_cpu[id] + n_i;
            for(int n_pid = 0; n_pid <  par_num_cpu[node_id]; n_pid++)
            {
                currentChildIDX_cpu[par_num_sum_cpu[node_start] + pid_total] 
                    = nets[net_idx].currentChildIDX_cpu[pid_total] + node_id * 10;
                par_nodes_cpu[par_num_sum_cpu[node_start] + pid_total] = nets[net_idx].par_nodes_cpu[pid_total] + node_cnt_sum_cpu[id];
                pid_total++;
            }
        }
        for(int i=node_cnt_sum_cpu[id];i<node_cnt_sum_cpu[id]+node_cnt_sum_cpu[id + 1];i++)
        {
            int depth_node_i = node_depth_cpu[i];
            depth_node_cnt_cpu[b_id*MAX_DEPTH+depth_node_i + 1]++;
        }
        if(false)
        {
            for(int i=node_cnt_sum_cpu[id];i<node_cnt_sum_cpu[id]+node_cnt_sum_cpu[id + 1];i++)
            {
                printf("node %d, %d children, %d fathers, depth %d\n", i, child_num_cpu[i], par_num_cpu[i], node_depth_cpu[i]);
                int curx = nodes_cpu[i]/ Y % X, cury = nodes_cpu[i] % Y;
                printf("    pos: %d, %d, %d\n", nodes_cpu[i]/ Y / X, curx, cury);
                printf("    parents: ");
                for(int p_id=0;p_id<par_num_cpu[i];p_id++)
                {
                    printf("%d:%d ", par_num_sum_cpu[i]+p_id, par_nodes_cpu[par_num_sum_cpu[i]+p_id]);
                }
                printf("    corresponding chidIdx: ");
                for(int p_id=0;p_id<par_num_cpu[i];p_id++)
                {
                    printf("%d ", currentChildIDX_cpu[par_num_sum_cpu[i]+p_id]);
                }
                printf("\n");
            }
    
            for(int i=node_cnt_sum_cpu[id];i<node_cnt_sum_cpu[id]+node_cnt_sum_cpu[id + 1];i++)
            {
                printf("node id: %d\n", i);
                int curx = nodes_cpu[i]/ Y % X, cury = nodes_cpu[i] % Y;
                printf("    pos: %d, %d\n", curx, cury);
                printf("    parents: ");
                for(int p_id=0;p_id<par_num_cpu[i];p_id++)
                {
                    printf("%d ", par_nodes_cpu[par_num_sum_cpu[i]+p_id]);
                }
                printf("\n");
            }
        }        
        node_cnt_sum_cpu[id + 1] += node_cnt_sum_cpu[id];
    }
    for(int bid=1; bid <= batches.size(); bid++)
    {
        batch_depth_cnt_cpu[bid]+=batch_depth_cnt_cpu[bid-1];
    }
    for(int bid=0; bid < batches.size(); bid++)
    {
        assert(batch_depth_cnt_cpu[bid + 1]-batch_depth_cnt_cpu[bid]<MAX_DEPTH);
        for(int d = 0; d< batch_depth_cnt_cpu[bid + 1]-batch_depth_cnt_cpu[bid]; d++)
        {
            depth_node_cnt_cpu[bid*MAX_DEPTH + d + 1]+=depth_node_cnt_cpu[bid*MAX_DEPTH+d];
        }
        for(int d = 0; d<= batch_depth_cnt_cpu[bid + 1]-batch_depth_cnt_cpu[bid]; d++)
        {
            depthID2nodeSequence[batch_depth_cnt_cpu[bid]+d] = node_cnt_sum_cpu[batch_cnt_sum[bid]] + depth_node_cnt_cpu[bid*MAX_DEPTH+d];
            depth_node_cnt_cpu[bid*MAX_DEPTH+d] += node_cnt_sum_cpu[batch_cnt_sum[bid]];
        }
        
        for(int node_id = node_cnt_sum_cpu[batch_cnt_sum[bid]]; node_id < node_cnt_sum_cpu[batch_cnt_sum[bid + 1]]; node_id++)
        {
            int depth = node_depth_cpu[node_id];
            depth_node_cpu[depth_node_cnt_cpu[bid*MAX_DEPTH+depth]++] = node_id;
        }
    }
    int node_total = node_cnt_sum_cpu[net_cnt];
    for(int node_id = 1; node_id <= node_total; node_id++)
    {
        child_num_sum_cpu[node_id] = child_num_cpu[node_id-1];
        child_num_sum_cpu[node_id] += child_num_sum_cpu[node_id - 1];
    }
    cudaMemcpy(net_ids, nets2route.data(), net_cnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(layer_output, 0xFF, reserve_node_num * sizeof(int));
    cudaMemcpy(node_cnt_sum, node_cnt_sum_cpu, (net_cnt + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nodes, nodes_cpu, node_cnt_sum_cpu[net_cnt] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(net_depth, net_depth_cpu, net_cnt * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(batch_depth, batch_depth_cnt_cpu, (batches.size() + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(child_num_sum, child_num_sum_cpu, (node_cnt_sum_cpu[net_cnt] + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(par_num_sum, par_num_sum_cpu, (node_total + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(node_net_idx, node_net_idx_cpu, node_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(node_depth, node_depth_cpu, node_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(depth_node, depth_node_cpu, node_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(par_nodes, par_nodes_cpu, par_num_sum_cpu[node_total] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(currentChildIDX, currentChildIDX_cpu, par_num_sum_cpu[node_total] * sizeof(int), cudaMemcpyHostToDevice);
    // int sequence_end;
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        if (t != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy Error: %s\n", cudaGetErrorString(t));
            exit(EXIT_FAILURE);
        }
    }
    init_costs<<<node_total, 10>>>(node_total*MAX_LAYER);
    init_min_child_costs<<<(child_num_sum_cpu[node_total] + 1) * MAX_LAYER, 1>>>((child_num_sum_cpu[node_total] + 1) * MAX_LAYER);
    init_road<<<(child_num_sum_cpu[node_total] + 1) * MAX_LAYER, 1>>>((child_num_sum_cpu[node_total] + 1) * MAX_LAYER);

    for(int i = 0; i < batches.size(); i++) {
        int total_node_num = batch_cnt_sum[i + 1] - batch_cnt_sum[i];
        int net_offset = batch_cnt_sum[i];
        int next_net_offset = batch_cnt_sum[i + 1];
        graph::commit_wire_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), 0, ++global_timestamp, -1, 0);
        graph::commit_via_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), 0, global_timestamp, -1, 0);     
        global_timestamp++;
        graph::update_cost();
        {
            cudaDeviceSynchronize();
            auto t = cudaGetLastError();
            if (t != cudaSuccess) {
                fprintf(stderr, "update_cost CUDA Error: %s\n", cudaGetErrorString(t));
                exit(EXIT_FAILURE);
            }
        }
        graph::compute_presum<<<all_track_cnt, THREAD_NUM, sizeof(double) * XY>>> ();
        {
            cudaDeviceSynchronize();
            auto t = cudaGetLastError();
            if (t != cudaSuccess) {
                fprintf(stderr, "compute_presum Error: %s\n", cudaGetErrorString(t));
                exit(EXIT_FAILURE);
            }
        }

        int cur_batch_depth = batch_depth_cnt_cpu[i + 1] - batch_depth_cnt_cpu[i];
        precompute_cost_edge<<<BLOCK_NUM(par_num_sum_cpu[node_total] * 9), THREAD_NUM>>>(par_num_sum_cpu[node_total]);
        {
            cudaDeviceSynchronize();
            auto t = cudaGetLastError();
            if (t != cudaSuccess) {
                fprintf(stderr, "precompute_cost_edge Error: %s\n", cudaGetErrorString(t));
                exit(EXIT_FAILURE);
            }
        }
        for(int d = cur_batch_depth - 1; d >= 0; d--)
        {
            int shift = depthID2nodeSequence[batch_depth_cnt_cpu[i]+d];
            int end_shift = depthID2nodeSequence[batch_depth_cnt_cpu[i] + d + 1];
            // Lshape_route_node_cuda<<<BLOCK_NUM(end_shift-shift + 1), 512>>> (shift, end_shift);
            Lshape_route_node_layer_change_cuda<<<BLOCK_NUM(end_shift-shift + 1), 512>>> (shift, end_shift);
            {
                cudaDeviceSynchronize();
                auto t = cudaGetLastError();
                if(t != 0) {
                    fprintf(stderr, "L shape route CUDA error %d\n", t);
                    exit(0);
                }
            }
        }
        for(int d = 0; d < cur_batch_depth; d++)
        {
            int shift = depthID2nodeSequence[batch_depth_cnt_cpu[i]+d];
            int end_shift = depthID2nodeSequence[batch_depth_cnt_cpu[i] + d + 1];
            // get_routing_tree_cuda<<<BLOCK_NUM(end_shift - shift + 1), 512>>> (shift, end_shift, d, global_timestamp);
            get_routing_tree_layer_change_cuda<<<BLOCK_NUM(end_shift-shift + 1), 512>>> (shift, end_shift, d, global_timestamp);
            {
                cudaDeviceSynchronize();
                auto t = cudaGetLastError();
                if(t != 0) {
                    fprintf(stderr, "get routing tree CUDA error %d\n", t);
                    exit(0);
                }
            }
        }
        graph::batch_wire_update(global_timestamp);
        graph::commit_via_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), 0, global_timestamp);
    }
}

}
#pragma once
#include "database.hpp"
#include <queue>
#include <stack>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
using namespace flute;

namespace cudb {

__managed__ int *rip_up_list;
__managed__ bool *congestion;
__managed__ float *congestion_xsum;
__managed__ float *congestion_ysum;
__managed__ int sparsity_gpu;
__managed__ int *original_net_pin_range;
std::mutex mtx;

__managed__ unsigned long long ROUTE_PER_PIN = 12; 
// TODO: fixd number of solution per pin may require large memory for complex cases. need to optimize this in the future.
__managed__ unsigned long long ROUTE_PER_PIN_PHASE2 = 100;

__managed__ int DIR, L, X, Y, XY, NET_NUM, PIN_NUM, PIN_NUM_PHASE2, *x_edge_len, *y_edge_len, *pin_acc_num, *pins;//DIR: direction of layer 0 in cuda database
__managed__ int *pin_acc_num_phase2;
__managed__ double unit_length_wire_cost, unit_via_cost, *unit_length_short_costs, *layer_min_len;
__managed__ float *vcost, *wcost, *capacity, *demand;
__managed__ int *cross_points;
__managed__ double total_wirelength = 0, total_overflow_cost, layer_overflow_cost[10];
__managed__ int total_via_count = 0;

__managed__ bool *is_of_net;
__managed__ int *of_edge_sum;
__managed__ int *routes, *timestamp, *pre_demand;
__managed__ int *routes_phase2;
__managed__ int *ripup_flag;
__managed__ int *last, all_track_cnt, *idx2track, *track_net, *track_pos, *track_xy;
__managed__ double *presum;
__managed__ int *net_ids;
int *net_ids_cpu;
int global_timestamp = 0;
vector<vector<int>> net_x_cpu, net_y_cpu;
vector<int> pin_cnt_sum_cpu;
vector<int> ripup_flag_cpu;
vector<int> pin_cnt_sum_phase2_cpu;

#define IDX(l, x, y) ((l) * X * Y + (x) * Y + (y))
#define THREAD_NUM 512
#define BLOCK_NUM(n) ((n) / THREAD_NUM + 1)
#define BLOCK_CNT(tot, thread_cnt) ((tot) / (thread_cnt) + ((tot) % (thread_cnt) > 0))
#define INF 1e22


void build_cuda_database();

struct net {
    void construct_rsmt(bool, bool);
    void generate_detours(bool* const & congestionView_cpu, 
                          float* const & congestionView_xsum_cpu, 
                          float* const & congestionView_ysum_cpu,
                          bool construct_segments = true, bool display = false);
    void generate_detours_reconstruct(bool* const & congestionView_cpu, 
                          float* const & congestionView_xsum_cpu, 
                          float* const & congestionView_ysum_cpu,
                          bool construct_segments = true, bool display = false, int net_id = 0);
    void calc_hpwl();
    
    int minx, maxx, miny, maxy, hpwl, original_net_id;
    vector<int> pins;
    vector<vector<int>> rsmt;
    vector<pair<int, int>> rsmt_h_segments, rsmt_v_segments;
    vector<int> par_num_cpu;
    vector<int> par_num_sum_cpu;
    vector<int> currentChildIDX_cpu;
    vector<int> par_nodes_cpu;
    vector<int> child_num_cpu;
    vector<int> node_depth_cpu;
    vector<int> nodes_cpu;
    vector<int> points;
    int node_index_cnt = 0;
    int MAX_LAYER=10;
    int select_root = 0;

    Tree tree;
    unordered_map<int, int> layer;
};
vector<net> nets;
vector<vector<net>> tmp_nets;


void net::calc_hpwl() {
    minx = X, maxx = 0, miny = Y, maxy = 0;
    for(auto p : pins) {
        int x = p / Y % X, y = p % Y;
        minx = min(minx, x);
        maxx = max(maxx, x);
        miny = min(miny, y);
        maxy = max(maxy, y);
    }
    hpwl = maxx - minx + maxy - miny;
}

__managed__ int *points;
__managed__ int *points_net_id;
__managed__ int points_total_gpu;
__managed__ int *points_num_sum;

__managed__ int *h_segments;
__managed__ int *h_segments_sum;

__managed__ int *v_segments;
__managed__ int *v_segments_sum;
__managed__ int h_segments_total_gpu, v_segments_total_gpu;

__managed__ int *compete_map;
__managed__ int *device_map;  // X * Y size
__managed__ int *batch_id;    // nets size
__managed__ int *net_status; //0: normal 1: candidate 2: occupied by other net
__managed__ bool have_commit; 
__managed__ bool have_commit2; 

__global__ void try_commit_kernel(int batch_id_to_try, int nets_count, int *points_num_sum, 
                                int *points, int *h_segments_sum, int *h_segments, 
                                int *v_segments_sum, int *v_segments, int X, int Y, int round_try) {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (net_id >= nets_count) return;
    if (net_status[net_id] == 2) return;  // already assigned
    if (batch_id[net_id] != -1) return;  // already assigned
    
    int success = 1;
    // Check points conflicts
    int start_idx = points_num_sum[net_id];
    int end_idx = points_num_sum[net_id + 1];
    
    // First pass: check existing assignments
    for (int i = start_idx; i < end_idx; i++) {
        int point = points[i];
        int pre_net = device_map[point];
        if (pre_net != -1 && batch_id[pre_net] != -1) {
            return;
        }
    }
    int h_start = h_segments_sum[net_id] * 2;
    int h_end = h_segments_sum[net_id + 1] * 2;
    for (int i = h_start; i < h_end; i += 2) {
        int start = h_segments[i];
        int end = h_segments[i + 1];
        assert(i+1 < h_segments_total_gpu*2);
        int x = start / Y;
        int y = start % Y;
        while (x <= end / Y) {
            atomicMax(&device_map[x * Y + y], net_id);
            x+=sparsity_gpu;
        }
    }
    
    int v_start = v_segments_sum[net_id] * 2;
    int v_end = v_segments_sum[net_id + 1] * 2;
    for (int i = v_start; i < v_end; i += 2) {
        int start = v_segments[i];
        int end = v_segments[i + 1];
        assert(i+1 < v_segments_total_gpu*2);
        int x = start / Y;
        int y = start % Y;
        while (y <= end % Y) {
            atomicMax(&device_map[x * Y + y], net_id);
            y+=sparsity_gpu;
        }
    }

    for (int i = start_idx; i < end_idx; i++) {
        int point = points[i];
        atomicMax(&device_map[point], net_id);
    }
    
    // Second pass: try to claim points
    for (int i = start_idx; i < end_idx; i++) {
        int point = points[i];
        if (device_map[point] != net_id) {
            success = 0;
            break;
        }
        if(success==0) break;
    }
    
    if (success) {
        // Mark segments
        int h_start = h_segments_sum[net_id] * 2;
        int h_end = h_segments_sum[net_id + 1] * 2;
        for (int i = h_start; i < h_end; i += 2) {
            int start = h_segments[i];
            int end = h_segments[i + 1];
            assert(i+1 < h_segments_total_gpu*2);
            int x = start / Y;
            int y = start % Y;
            while (x <= end / Y) {
                atomicMax(&device_map[x * Y + y], net_id);
                x+=sparsity_gpu;
            }
        }
        
        int v_start = v_segments_sum[net_id] * 2;
        int v_end = v_segments_sum[net_id + 1] * 2;
        for (int i = v_start; i < v_end; i += 2) {
            int start = v_segments[i];
            int end = v_segments[i + 1];
            assert(i+1 < v_segments_total_gpu*2);
            int x = start / Y;
            int y = start % Y;
            while (y <= end % Y) {
                int pre_net_id = device_map[x * Y + y];
                if(pre_net_id < net_id && net_status[pre_net_id] == 0)
                {
                    net_status[pre_net_id] = 2;
                }
                atomicMax(&device_map[x * Y + y], net_id);
                // device_map[x * Y + y] = net_id;
                y+=sparsity_gpu;
            }
        }
        
        
        batch_id[net_id] = batch_id_to_try;
        have_commit = true;
        have_commit2 = true;
    }
}

__global__ void revert_low_priority(int batch_id_to_try, int nets_count, int *points_num_sum, 
                                int *points, int *h_segments_sum, int *h_segments, 
                                int *v_segments_sum, int *v_segments, int X, int Y, int round_try) {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (net_id >= nets_count) return;
    if (net_status[net_id] != 2) return;
    
    int start_idx = points_num_sum[net_id];
    int end_idx = points_num_sum[net_id + 1];

    // try to occupy points
    int h_start = h_segments_sum[net_id] * 2;
    int h_end = h_segments_sum[net_id + 1] * 2;
    for (int i = h_start; i < h_end; i += 2) {
        assert(i+1 < h_segments_total_gpu*2);
        int start = h_segments[i];
        int end = h_segments[i + 1];
        int x = start / Y;
        int y = start % Y;
        assert(x>=0&&x<X&&y>=0&&y<Y);
        while (x <= end / Y) {
            if(device_map[x * Y + y]==net_id)
            {
                device_map[x * Y + y] = -1;
            }
            x+=sparsity_gpu;
        }
    }
    
    int v_start = v_segments_sum[net_id] * 2;
    int v_end = v_segments_sum[net_id + 1] * 2;
    for (int i = v_start; i < v_end; i += 2) {
        assert(i+1 < v_segments_total_gpu*2);
        int start = v_segments[i];
        int end = v_segments[i + 1];
        int x = start / Y;
        int y = start % Y;
        assert(x>=0&&x<X&&y>=0&&y<Y);
        while (y <= end % Y) {
            if(device_map[x * Y + y]==net_id)
            {
                device_map[x * Y + y] = -1;
            }
            y+=sparsity_gpu;
        }
    }

    for (int i = start_idx; i < end_idx; i++) {
        int point = points[i];
        if(device_map[point]==net_id)
        {
            device_map[point] = -1;
        }
        // atomicMax(&device_map[point], net_id);
    }
}

__global__ void compete_map_kernel(int batch_id_to_try, int nets_count, int *points_num_sum, 
                                int *points, int *h_segments_sum, int *h_segments, 
                                int *v_segments_sum, int *v_segments, int X, int Y, int round_try) {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (net_id >= nets_count) return;
    if (net_status[net_id] == 2) return;  // already assigned
    if (batch_id[net_id] != -1) return;  // already assigned
    
    int start_idx = points_num_sum[net_id];
    int end_idx = points_num_sum[net_id + 1];

    net_status[net_id] = 1;

    int xs[4] = {0,0,-1,1};
    int ys[4] = {0,-1,0,0};
    for (int i = start_idx; i < end_idx; i++) {
        int point = points[i];
        int x = point / Y;
        int y = point % Y;
        for(int j=0; j<4; j++)
        {
            int x_new = x + xs[j];
            int y_new = y + ys[j];
            if(x_new>=0 && x_new<X && y_new>=0 && y_new<Y)
            {
                atomicMax(&device_map[x_new * Y + y_new], net_id);
                if(j==0)
                if(device_map[x_new * Y + y_new] != net_id)
                {
                    return;
                }
            }
        }
    }

    int h_start = h_segments_sum[net_id] * 2;
    int h_end = h_segments_sum[net_id + 1] * 2;
    for (int i = h_start; i < h_end; i += 2) {
        int start = h_segments[i];
        int end = h_segments[i + 1];
        assert(i+1 < h_segments_total_gpu*2);
        assert(start/Y==end/Y||start%Y==end%Y);
        int x = start / Y;
        int y = start % Y;
        while (x <= end / Y) {
            atomicMax(&device_map[x * Y + y], net_id);
            x+=sparsity_gpu;
        }
    }
    
    int v_start = v_segments_sum[net_id] * 2;
    int v_end = v_segments_sum[net_id + 1] * 2;
    for (int i = v_start; i < v_end; i += 2) {
        int start = v_segments[i];
        int end = v_segments[i + 1];
        assert(i+1 < v_segments_total_gpu*2);
        int x = start / Y;
        int y = start % Y;
        assert(start/Y==end/Y||start%Y==end%Y);
        
        while (y <= end % Y) {
            atomicMax(&device_map[x * Y + y], net_id);
            y+=sparsity_gpu;
        }
    }
}


__global__ void compete_map_kernel_commit(int batch_id_to_try, int nets_count, int *points_num_sum, 
                                int *points, int *h_segments_sum, int *h_segments, 
                                int *v_segments_sum, int *v_segments, int X, int Y, int round_try) {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (net_id >= nets_count) return;
    if (net_status[net_id] == 2) return;  // already assigned
    if (batch_id[net_id] != -1) return;  // already assigned
    
    int success = 1;
    int start_idx = points_num_sum[net_id];
    int end_idx = points_num_sum[net_id + 1];
    
    for (int i = start_idx; i < end_idx; i++) {
        int point = points[i];
        if (device_map[point] != net_id) {
            success = 0;
            break;
        }
        if(success==0) break;
    }

    if (success) {
        batch_id[net_id] = batch_id_to_try;
        have_commit = true;
        have_commit2 = true;
    }
}

__global__ void revert_unsuccess(int batch_id_to_try, int nets_count, int *points_num_sum, 
                                int *points, int *h_segments_sum, int *h_segments, 
                                int *v_segments_sum, int *v_segments, int X, int Y, int round_try) {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (net_id >= nets_count) return;
    if (net_status[net_id] != 1) return;  // already assigned
    if (batch_id[net_id] != -1) return;
    
    // Check points conflicts
    int start_idx = points_num_sum[net_id];
    int end_idx = points_num_sum[net_id + 1];
    
    // First pass: check existing assignments
    for (int i = start_idx; i < end_idx; i++) {
        int point = points[i];
        int pre_net = device_map[point];
        if (pre_net!=-1 && batch_id[pre_net] != -1) {
            net_status[net_id] = 2;
            return;
        }
    }
}

__global__ void revert_unsuccess_v2() { // try to use X*Y threads
    int point_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_id >= points_total_gpu) return;
    int net_id = points_net_id[point_id];
    if (net_status[net_id] != 1) return;  // already assigned
    if (batch_id[net_id] != -1) return;

    int pre_net = device_map[points[point_id]];
    if(pre_net!=-1 && batch_id[pre_net]!=-1 && net_status[net_id]!=2)
    {
        net_status[net_id] = 2;
    }
}

__global__ void revert_unsuccess_phase2() {
    int pos_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos_id >= X*Y) return;
    if(device_map[pos_id] < 0)
    {
        return;
    }
    int net_id = device_map[pos_id];
    if(net_status[net_id] == 2)
    {
        device_map[pos_id] = -1;
    }
    return;
}


vector<vector<int>> generate_batches_rsmt_gpu(vector<int> &nets2route_tmp, int MAX_BATCH_SIZE = 1000000) { // exist bug. not deterministic currently   
    auto _time = elapsed_time();
    sparsity_gpu = 15;
    if(mode==1||mode==2)
    {
        sparsity_gpu=2;
    }
    vector<int> nets2route(nets2route_tmp.size(), 0);
    for(int i = 0; i < nets2route_tmp.size(); i++) {
        nets2route[i] = nets2route_tmp[nets2route_tmp.size()-1-i];
    }
    cudaMalloc(&device_map, X * Y * sizeof(int));
    cudaMalloc(&compete_map, X * Y * sizeof(int));
    cudaMalloc(&batch_id, nets2route.size() * sizeof(int));
    cudaMalloc(&net_status, nets2route.size() * sizeof(int));
    cudaMemset(net_status, 0, nets2route.size());

    cudaMalloc(&points_num_sum, (nets2route.size()+1) * sizeof(int));
    
    vector<int> points_all;
    vector<int> points_net_all;
    vector<int> h_segments_all;
    vector<int> v_segments_all;
    
    vector<int> points_sum_cpu(nets2route.size()+1, 0);
    vector<int> h_segments_sum_cpu(nets2route.size()+1, 0);
    vector<int> v_segments_sum_cpu(nets2route.size()+1, 0);
    
    int points_total = 0;
    int h_segments_total = 0;
    int v_segments_total = 0;
    
    // Calculate cumulative sums
    for(int i = 0; i < nets2route.size(); i++){
        int net_id = nets2route[i];
        points_total += nets[net_id].points.size();
        h_segments_total += nets[net_id].rsmt_h_segments.size();
        v_segments_total += nets[net_id].rsmt_v_segments.size();
        
        points_sum_cpu[i+1] = points_sum_cpu[i] + nets[net_id].points.size();
        h_segments_sum_cpu[i+1] = h_segments_sum_cpu[i] + nets[net_id].rsmt_h_segments.size();
        v_segments_sum_cpu[i+1] = v_segments_sum_cpu[i] + nets[net_id].rsmt_v_segments.size();
    }
    
    // Resize vectors
    points_all.resize(points_total);
    points_total_gpu = points_total;
    points_net_all.resize(points_total);
    h_segments_all.resize(h_segments_total * 2); // Each segment has 2 endpoints
    v_segments_all.resize(v_segments_total * 2);
    h_segments_total_gpu = h_segments_total;
    v_segments_total_gpu = v_segments_total;
    // Fill data
    int p_id = 0, h_id = 0, v_id = 0;
    for(int nid = 0; nid < nets2route.size(); nid++) {
        int net_id = nets2route[nid];
        for(auto p : nets[net_id].points) {
            points_net_all[p_id] = nid;
            points_all[p_id++] = p;
        }
        for(auto h_seg : nets[net_id].rsmt_h_segments) {
            h_segments_all[h_id++] = h_seg.first;
            h_segments_all[h_id++] = h_seg.second;
        }
        for(auto v_seg : nets[net_id].rsmt_v_segments) {
            v_segments_all[v_id++] = v_seg.first;
            v_segments_all[v_id++] = v_seg.second;
        }
    }
    assert(h_segments_total * 2 == h_id);
    assert(v_segments_total * 2 == v_id);
    
    // Allocate and copy to device
    cudaMalloc(&h_segments, h_segments_total * 2 * sizeof(int));
    cudaMalloc(&v_segments, v_segments_total * 2 * sizeof(int));
    cudaMalloc(&h_segments_sum, (nets2route.size()+1) * sizeof(int));
    cudaMalloc(&v_segments_sum, (nets2route.size()+1) * sizeof(int));
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        if (t != cudaSuccess) {
            fprintf(stderr, "cudaMalloc v_segments_sum Error: %s\n", cudaGetErrorString(t));
            exit(EXIT_FAILURE);
        }
    }
    cudaMalloc(&points, points_total * sizeof(int));
    cudaMalloc(&points_net_id, points_total * sizeof(int));
    // Copy data to device
    cudaMemcpy(points_num_sum, points_sum_cpu.data(), (nets2route.size()+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(points, points_all.data(), points_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(points_net_id, points_net_all.data(), points_total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h_segments_sum, h_segments_sum_cpu.data(), (nets2route.size()+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(v_segments_sum, v_segments_sum_cpu.data(), (nets2route.size()+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(h_segments, h_segments_all.data(), h_segments_total * 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(v_segments, v_segments_all.data(), v_segments_total * 2 * sizeof(int), cudaMemcpyHostToDevice);

    vector<int> init_batch_id(nets2route.size(), -1);
    cudaMemcpy(batch_id, init_batch_id.data(), nets2route.size() * sizeof(int), cudaMemcpyHostToDevice);
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        if (t != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy batch_id Error: %s\n", cudaGetErrorString(t));
            exit(EXIT_FAILURE);
        }
    }

    int current_batch = 0;
    const int BLOCK_SIZE = 256;
    int num_blocks = (nets2route.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    while (true) {
        have_commit = false;
        // Initialize device_map to -1
        cudaMemset(device_map, -1, X * Y * sizeof(int));
        cudaMemset(compete_map, -1, X * Y * sizeof(int));
        cudaMemset(net_status, 0, nets2route.size() * sizeof(int));
        // Try 3 rounds for each batch
        for (int round = 0; round < 30; round++) {
            have_commit2 = false;
            compete_map_kernel<<<num_blocks, BLOCK_SIZE>>>(
                current_batch, nets2route.size(), 
                points_num_sum, points,
                h_segments_sum, h_segments,
                v_segments_sum, v_segments,
                X, Y, round);
            {
                cudaDeviceSynchronize();
                auto t = cudaGetLastError();
                if (t != cudaSuccess) {
                    fprintf(stderr, "compete_map_kernel CUDA Error: %s\n", cudaGetErrorString(t));
                    exit(EXIT_FAILURE);
                }
            }            
            compete_map_kernel_commit<<<num_blocks, BLOCK_SIZE>>>(
                current_batch, nets2route.size(), 
                points_num_sum, points,
                h_segments_sum, h_segments,
                v_segments_sum, v_segments,
                X, Y, round);
            {
                cudaDeviceSynchronize();
                auto t = cudaGetLastError();
                if (t != cudaSuccess) {
                    fprintf(stderr, "compete_map_kernel_commit CUDA Error: %s\n", cudaGetErrorString(t));
                    exit(EXIT_FAILURE);
                }
            }
            
            if(!have_commit2) break;
            revert_unsuccess_v2<<<(points_total+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>();
            {
                cudaDeviceSynchronize();
                auto t = cudaGetLastError();
                if (t != cudaSuccess) {
                    fprintf(stderr, "revert_unsuccess_v2 Error: %s\n", cudaGetErrorString(t));
                    exit(EXIT_FAILURE);
                }
            }
            revert_unsuccess_phase2<<<(X*Y+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>();
            {
                cudaDeviceSynchronize();
                auto t = cudaGetLastError();
                if (t != cudaSuccess) {
                    fprintf(stderr, "revert_unsuccess_phase2 CUDA Error: %s\n", cudaGetErrorString(t));
                    exit(EXIT_FAILURE);
                }
            }
        }
        
        if (!have_commit) break;
        current_batch++;
    }
    // Copy results back to CPU
    vector<int> final_batch_id(nets2route.size());
    cudaMemcpy(final_batch_id.data(), batch_id, nets2route.size() * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Convert to return format
    vector<vector<int>> batches(current_batch);
    for (int i = 0; i < nets2route.size(); i++) {
        assert(final_batch_id[i]!=-1);
        if (final_batch_id[i] != -1) {
            batches[final_batch_id[i]].push_back(nets2route[i]);
        }
    }
    
    // Free GPU memory
    cudaFree(device_map);
    cudaFree(batch_id);

    _time = elapsed_time() - _time;
    if(LOG) cout << setw(40) << "Batch" << setw(20) << "#Nets" << setw(20) << "#Batches" << setw(20) << "Time" << endl;
    if(LOG) cout << setw(40) << "Generation" << setw(20) << nets2route.size() << setw(20) << batches.size() << setw(20) << setprecision(2) << _time << endl;
    return batches;
}


vector<vector<int>> generate_batches_rsmt(vector<int> &nets2route, int MAX_BATCH_SIZE = 1000000) {    
    auto _time = elapsed_time();
    vector<vector<int>> batches;
    vector<vector<bool>> batch_vis;
    // vector<int> lookupTable(X * Y, 0);
   
    auto has_conflict = [&] (int net_id, int batch_id) {

        for(auto p : nets[net_id].points) 
        {
            assert(p < X * Y);
            if(batch_vis[batch_id][p])
            {
                return true;
            }
        }
        return false;
    };

    auto mark_3x3 = [&] (int pos, int batch_id) {
        int _x = pos / Y, _y = pos % Y;
        for(int x = _x - 1; x <= _x + 1; x++) if(0 <= x && x < X)
            for(int y = _y - 1; y < _y + 1; y++) if(0 <= y && y < Y)
                if(x == _x || y == _y) batch_vis[batch_id][x * Y + y] = 1;
    };


    long long segment_len = 0, segment_cnt = 0, failed = 0;
    for(auto net_id : nets2route) {
        int batch_id = -1;
        for(int i = 0; i < batches.size(); i++) if(batches[i].size() < MAX_BATCH_SIZE)
        {
            if(!has_conflict(net_id, i)) { batch_id = i; break; }
            else failed++;
        }
        if(batch_id == -1) {
            batch_id = batches.size();
            batches.emplace_back(vector<int> ());
            batch_vis.emplace_back(vector<bool> (X * Y, 0));
        }
        batches[batch_id].emplace_back(net_id);
        int sparsity = 20;
        if(mode==1||mode==2) sparsity = 5; 
        for(auto seg : nets[net_id].rsmt_h_segments) {
            segment_len += seg.second / Y - seg.first / Y;
            segment_cnt++;
            for(auto p = seg.first; p <= seg.second; p += sparsity * Y) batch_vis[batch_id][p] = 1;
        }
        for(auto seg : nets[net_id].rsmt_v_segments) {
            for(auto p = seg.first; p <= seg.second; p += sparsity) batch_vis[batch_id][p] = 1;
        }
        for(auto p : nets[net_id].points) mark_3x3(p, batch_id);
    }
    _time = elapsed_time() - _time;
    printf("INFO: AVG LEN = %lld/%lld = %.2f\n", segment_len,segment_cnt, 1.0 * segment_len / segment_cnt);
    printf("INFO: failed = %d\n", failed);
    if(LOG) cout << setw(40) << "Batch" << setw(20) << "#Nets" << setw(20) << "#Batches" << setw(20) << "Time" << endl;
    if(LOG) cout << setw(40) << "Generation" << setw(20) << nets2route.size() << setw(20) << batches.size() << setw(20) << setprecision(2) << _time << endl;
    return move(batches);
}

double TOT_RSMT_LENGTH = 0;
vector<vector<int>> my_flute(unordered_set<int> &pos) {
    vector<int> x;
    vector<int> y;
    int cnt = 0;
    vector<int> nodes, parent;
    vector<tuple<int, int, int>> edges;
    x.reserve(pos.size());
    y.reserve(pos.size());
    for(auto e : pos) {
        x.emplace_back(db::dr_x[e / Y]);
        y.emplace_back(db::dr_y[e % Y]);
        cnt++;
    }
    auto tree = flute::flute(x, y, 6);
    for(int i = 0; i < cnt * 2 - 2; i++) {
        Branch &branch = tree.branch[i];
        nodes.emplace_back(db::dr2gr_x[branch.x] * Y + db::dr2gr_y[branch.y]);
    }
    sort(nodes.begin(), nodes.end());
    nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());
    parent.resize(nodes.size());
    for(int i = 0; i < nodes.size(); i++) parent[i] = i;
    edges.reserve(cnt * 2);
    for(int i = 0; i < cnt * 2 - 2; i++) if(tree.branch[i].n < cnt * 2 - 2) {
        Branch &branch1 = tree.branch[i], &branch2 = tree.branch[branch1.n];
        int u, v;
        u = lower_bound(nodes.begin(), nodes.end(), db::dr2gr_x[branch1.x] * Y + db::dr2gr_y[branch1.y]) - nodes.begin();
        v = lower_bound(nodes.begin(), nodes.end(), db::dr2gr_x[branch2.x] * Y + db::dr2gr_y[branch2.y]) - nodes.begin();
        if(u == v) continue;
        edges.emplace_back(make_tuple(abs(branch1.x - branch2.x) + abs(branch1.y - branch2.y), u, v));
            
    }
    sort(edges.begin(), edges.end());
    function<int(int)> find_parent = [&] (int x) { return x == parent[x] ? x : parent[x] = find_parent(parent[x]); };
    vector<vector<int>> graph(nodes.size());
    for(auto edge : edges) {
        int u = get<1> (edge), v = get<2> (edge), par_u = find_parent(u), par_v = find_parent(v);
        if(par_u != par_v) {
            graph[u].emplace_back(v);
            graph[v].emplace_back(u);
            TOT_RSMT_LENGTH += get<0> (edge);
            parent[par_u] = par_v;
        }
    }
    int tot_degree = 0;
    for(int i = 0; i < nodes.size(); i++) tot_degree += graph[i].size();
    graph.emplace_back(move(nodes));
    return move(graph);
}

void net::construct_rsmt(bool rsmt_only = false, bool log_flag = false) {
    unordered_map<int, int> layer;
    unordered_set<int> pos, nodes;
    for(int i = 0; i < pins.size(); i++) {
        int pos2D = pins[i] % (X * Y);
        pos.insert(pos2D);
        layer[pos2D] = pins[i] / X / Y;
    }
    assert(pos.size() == pins.size());
    auto tmp = rsmt;
    if(!rsmt.size())
        rsmt = my_flute(pos);
    else {
        for(auto &e : rsmt.back()){
            e %= X * Y;
        }
    }
 
    if(!rsmt_only){
        rsmt_h_segments.clear();
        rsmt_v_segments.clear();
        points.clear();
        rsmt_h_segments.reserve(rsmt.size());
        rsmt_v_segments.reserve(rsmt.size());
        points = rsmt.back();
        for(int i = 0; i < rsmt.back().size(); i++) {
            int xi = rsmt.back()[i] / Y, yi = rsmt.back()[i] % Y;
            for(auto j : rsmt[i]) if(j < i) {
                int xj = rsmt.back()[j] / Y, yj = rsmt.back()[j] % Y;
                int minx = min(xi, xj), maxx = max(xi, xj), miny = min(yi, yj), maxy = max(yi, yj);
                if(xi != xj && yi != yj) {
                    rsmt_h_segments.emplace_back(minx * Y + miny, maxx * Y + miny);
                    rsmt_h_segments.emplace_back(minx * Y + maxy, maxx * Y + maxy);
                    rsmt_v_segments.emplace_back(minx * Y + miny, minx * Y + maxy);
                    rsmt_v_segments.emplace_back(maxx * Y + miny, maxx * Y + maxy);
                    points.emplace_back(xi * Y + yj);
                    points.emplace_back(xj * Y + yi);
                } else if(xi != xj) {
                    rsmt_h_segments.emplace_back(minx * Y + miny, maxx * Y + miny);
                    points.emplace_back(minx * Y + miny);
                    points.emplace_back(maxx * Y + miny);
                } else if(yi != yj) {
                    rsmt_v_segments.emplace_back(minx * Y + miny, minx * Y + maxy);
                    points.emplace_back(minx * Y + miny);
                    points.emplace_back(minx * Y + maxy);
                } else {
                    cerr << "error" << endl;
                }
            }
        }
    }
    for(auto &e : rsmt.back()){
        e += (layer.count(e) ? layer[e] : L) * X * Y;
    }

    if(log_flag){
        for(auto &e : tmp.back()) 
            e += (layer.count(e) ? layer[e] : L) * X * Y;
        printf("(\n");
        for(int i = 0;i < tmp.back().size(); i++){
            for(auto j : tmp[i]){
                printf("%d %d %d %d %d %d\n", tmp.back()[i] / Y % X, tmp.back()[i] % Y, tmp.back()[i] / Y / X, tmp.back()[j] / Y % X, tmp.back()[j] % Y, tmp.back()[j] / Y / X);
            }
        }
        printf(")\n");
        printf("(\n");

        for(int i = 0;i < rsmt.back().size(); i++){
            for(auto j : rsmt[i]){
                printf("%d %d %d %d %d %d\n", rsmt.back()[i] / Y % X, rsmt.back()[i] % Y, rsmt.back()[i] / Y / X, rsmt.back()[j] / Y % X, rsmt.back()[j] % Y, rsmt.back()[j] / Y / X);
            }
        }
        printf(")\n");
    }
}


typedef unsigned int BITSET_TYPE;
const int BITSET_LEN = 32;

int select_root_net(vector<vector<int>> rsmt)
{
    queue<pair<int,int>> list;
    int visited[rsmt.size()-1];
    int select = -1;
    for(int i=0; i < rsmt.size()-1; i++)
    {
        visited[i]=0;
        if(rsmt[i].size()==1)
        {
            list.push(make_pair(i, -1));
        }
    }
    while(!list.empty())
    {
        pair<int,int> front_element = list.front();
        list.pop();
        select = front_element.first;
        if(visited[select]) continue;
        visited[select]=1;
        int fa = front_element.second;
        for(int j = 0; j< rsmt[select].size(); j++)
        {
            if(rsmt[select][j]!=fa)
            {
                list.push(make_pair(rsmt[select][j], select));
            }
        }
    }
    return select;
}

bool compareIntervals(const std::pair<int, int> &a, const std::pair<int, int> &b) {
    if(a.first == b.first) return a.second < b.second;
    return a.first < b.first;
}

void mergeIntervals(std::vector<std::pair<int, int>> &intervals) {
    if (intervals.empty()) {
        return; // If the vector is empty, there's nothing to merge
    }

    // Sort the intervals based on the startting position
    std::sort(intervals.begin(), intervals.end(), compareIntervals);

    std::vector<std::pair<int, int>> merged; // This will store the merged intervals

    // startt by adding the first interval to the merged list
    merged.push_back(intervals[0]);

    for (const auto &interval : intervals) {
        // If the current interval's startt is less than or equal to the end of the last merged interval,
        // then we have an overlap, so we merge them
        if (interval.first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, interval.second);
        } else {
            // If there is no overlap, simply add the current interval to the merged list
            merged.push_back(interval);
        }
    }

    // Copy the merged intervals back to the original vector
    intervals = move(merged);
}

void net::generate_detours(bool* const & congestionView_cpu, 
                          float* const & congestionView_xsum_cpu, 
                          float* const & congestionView_ysum_cpu,
                          bool construct_segments, 
                          bool display) {
    
    auto graph_x = rsmt;
    int node_estimate = (graph_x.size()-1)*11;

    par_num_cpu.clear();
    par_num_sum_cpu.clear();
    currentChildIDX_cpu.clear();
    par_nodes_cpu.clear();
    child_num_cpu.clear();
    node_depth_cpu.clear();
    nodes_cpu.clear();
    node_index_cnt = 0;

    par_num_cpu.reserve(node_estimate);
    par_num_sum_cpu.reserve(node_estimate);
    currentChildIDX_cpu.reserve(node_estimate);
    par_nodes_cpu.reserve(node_estimate);
    child_num_cpu.reserve(node_estimate);
    node_depth_cpu.reserve(node_estimate);
    nodes_cpu.reserve(node_estimate);

    par_num_cpu.emplace_back(0);
    par_num_sum_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    currentChildIDX_cpu.emplace_back(0);
    par_nodes_cpu.emplace_back(0);
    child_num_cpu.emplace_back(0);
    node_depth_cpu.emplace_back(0);
    nodes_cpu.emplace_back(0);
    
    int depth_max = 0;
    select_root = select_root_net(graph_x);
    vector<int> congestionRegionID[2];
    vector<vector<pair<int, int> >> congestionRanges;
    congestionRanges.resize(2);
    vector<vector<vector<int>>> stems;
    stems.resize(2);
    congestionRegionID[0].resize(graph_x.size());
    congestionRegionID[1].resize(graph_x.size());
    for(int g_i = 0; g_i < graph_x.size(); g_i++)
    {
        congestionRegionID[0][g_i] = -1;
        congestionRegionID[1][g_i] = -1;
    }
    congestionRanges[0].resize(graph_x.size());
    congestionRanges[1].resize(graph_x.size());
    stems[0].resize(graph_x.size());
    stems[1].resize(graph_x.size());
    function<int(int, int)> getRegionID = [&] (int x, int direction) {
        if(x==-1) return -1;
        if(congestionRegionID[direction][x] == -1) return -1;
        if(congestionRegionID[direction][x] != x)
        {
            if(congestionRegionID[direction][x] == x)
            {
                assert(0);
            }
            int ans = getRegionID(congestionRegionID[direction][x], direction);
            congestionRegionID[direction][x] = ans;
            return ans;
        }else{
            return congestionRegionID[direction][x];
        }
    };
    for(int x = 0; x < graph_x.back().size(); x++)
    {
        int position_cur = graph_x.back()[x];
        int curl = position_cur / Y /X, curx = position_cur / Y % X, cury = position_cur % Y;
        for(int dir=0; dir<2; dir++)
        {
            if(curl<MAX_LAYER-1)
            {
                int stem_pos = dir?curx:cury;
                stems[dir][x].push_back(stem_pos);
            }
            int trunk_pos = dir?cury:curx;
            congestionRanges[dir][x] = make_pair(trunk_pos, trunk_pos);
        }
     
    }
    function<void(int, int)> markCongestion = [&] (int x, int par) {
        int position_cur = graph_x.back()[x];
        int curx = position_cur / Y % X, cury = position_cur % Y;
        for(auto e : graph_x[x]) if(e != par)
        {
            int ex = graph_x.back()[e] / Y % X;
            int ey = graph_x.back()[e] % Y;
            int dir=-1;
            int congestion = -1;
            if(ex == curx)
            {
                bool is_congestion_y = (congestionView_ysum_cpu[curx*Y+max(ey, cury)]-congestionView_ysum_cpu[curx*Y+min(ey, cury)])>0;
                if(is_congestion_y)
                {
                    dir = 1;
                    congestion = 1;
                }
            }
            else if(ey == cury)
            {
                bool is_congestion_x = (congestionView_xsum_cpu[max(ex, curx)*Y+cury]-congestionView_xsum_cpu[min(ex, curx)*Y+cury])>0;
                if(is_congestion_x)
                {
                    dir = 0;
                    congestion = 1;
                }
            }
            else{
                bool is_congestion_y = (congestionView_ysum_cpu[curx*Y+max(ey, cury)]-congestionView_ysum_cpu[curx*Y+min(ey, cury)])>0 || 
                                        (congestionView_ysum_cpu[ex*Y+max(ey, cury)]-congestionView_ysum_cpu[ex*Y+min(ey, cury)])>0;
                if(is_congestion_y)
                {
                    dir = 1;
                    congestion = 2;
                }
                if(congestion==-1)
                {

                    bool is_congestion_x = (congestionView_xsum_cpu[max(ex, curx)*Y+cury]-congestionView_xsum_cpu[min(ex, curx)*Y+cury])>0 || 
                                        (congestionView_xsum_cpu[max(ex, curx)*Y+ey]-congestionView_xsum_cpu[min(ex, curx)*Y+ey])>0;
                    if(is_congestion_x)
                    {
                        dir = 0;
                        congestion = 2;
                    }
                }
            }
            if(congestion == 1)
            {
                int target_region = -1;
                if(x!=select_root)
                {
                    if(congestionRegionID[dir][x]==-1)
                    {
                        congestionRegionID[dir][x] = x;
                    }
                    int region_x = getRegionID(x, dir);
                    target_region = region_x;
                }else{
                    if(congestionRegionID[dir][e]==-1)
                    {
                        congestionRegionID[dir][e] = e;
                    }
                    int region_e = getRegionID(e, dir);
                    target_region = region_e;
                }
                
                congestionRegionID[dir][e] = target_region;
                if(x!=select_root)
                for(auto pos: stems[dir][e])
                {
                    stems[dir][target_region].push_back(pos);
                }
                congestionRanges[dir][target_region].first = min(congestionRanges[dir][target_region].first, congestionRanges[dir][e].first);
                congestionRanges[dir][target_region].second = max(congestionRanges[dir][target_region].second, congestionRanges[dir][e].second);
            }
            else if(congestion == 2)
            {
                if(x!=select_root&&congestionRegionID[dir][x]==-1)
                {
                    congestionRegionID[dir][x] = x;
                    
                }
                if(congestionRegionID[dir][e]==-1)
                {
                    congestionRegionID[dir][e] = e;
                }
            }
            markCongestion(e, x);
        }
        for(int dir=0; dir<2; dir++)
        {
            int x_region = getRegionID(x, dir);
            for(auto e : graph_x[x]) if(e != par)
            {
                int ex = graph_x.back()[e] / Y % X;
                int ey = graph_x.back()[e] % Y;
                int e_region = getRegionID(e, dir);
                if(x_region!=e_region)
                {
                    if(x_region>=0)
                    {
                        stems[dir][x_region].push_back(dir?ex:ey);
                        congestionRanges[dir][x_region].first = min(congestionRanges[dir][x_region].first, dir?ey:ex);
                        congestionRanges[dir][x_region].second = max(congestionRanges[dir][x_region].second, dir?ey:ex);
                    }
                    if(e_region>=0)
                    {
                        stems[dir][e_region].push_back(dir?curx:cury);
                        congestionRanges[dir][e_region].first = min(congestionRanges[dir][e_region].first, dir?cury:curx);
                        congestionRanges[dir][e_region].second = max(congestionRanges[dir][e_region].second, dir?cury:curx);
                    }
                }
            }
        }
    };
    markCongestion(select_root, -1);

    function<int(int, int, int, int)> create_node = [&] (int l, int x, int y, int num_child) {
        int node_idx_insert = node_index_cnt++;// modify to 0; startt from zero
        par_num_cpu.emplace_back(0);
        par_num_sum_cpu.emplace_back(0);
        par_nodes_cpu.emplace_back(0);
        child_num_cpu.emplace_back(0);
        node_depth_cpu.emplace_back(0);
        nodes_cpu.emplace_back(0);
        
        par_num_sum_cpu[node_idx_insert+1] = 0;
        node_depth_cpu[node_idx_insert] = 0;
        child_num_cpu[node_idx_insert] = num_child;
        nodes_cpu[node_idx_insert] = l * X * Y + x * Y + y;
        return node_idx_insert;
    };

    function<int(int, int, int)> connect_node = [&] (int par_node_index, int cur_index, int cur_child_id) {
        int node_idx_insert = cur_index;
        int position_cur = nodes_cpu[cur_index];
        int curx = position_cur/ Y % X, cury = position_cur % Y;
        int position_par = nodes_cpu[par_node_index];
        int parx = position_par/ Y % X, pary = position_par % Y;
        if(construct_segments)
        {
            if(curx==parx)
            {
                rsmt_v_segments.emplace_back(make_pair(curx*Y+min(cury, pary), curx*Y+max(cury, pary)));
            }
            if(cury==pary)
            {
                rsmt_h_segments.emplace_back(make_pair(min(curx, parx)*Y+cury, max(curx, parx)*Y+cury));
            }
        }        
        assert(curx==parx||cury==pary);
        node_depth_cpu[node_idx_insert] = max(node_depth_cpu[node_idx_insert], node_depth_cpu[par_node_index] + 1);
        int depth = node_depth_cpu[node_idx_insert];

        int position = par_num_sum_cpu[node_idx_insert]+par_num_cpu[node_idx_insert]++;
        par_nodes_cpu.emplace_back(0);
        currentChildIDX_cpu.emplace_back(0);
        par_nodes_cpu[position] = par_node_index;
        depth_max = max(depth_max, depth+1);
        assert(cur_child_id<child_num_cpu[par_node_index]);
        currentChildIDX_cpu[position]=cur_child_id;
        return node_idx_insert;
    };

    float ratio = 0.10;
    int num_tracks = 7;
    function<int(int, int, int)> calc_displace = [&] (int query_pos, int dir, int region_id) {
        int ans = 0;
        for(auto pos: stems[dir][region_id])
        {
            ans+=abs(pos-query_pos);
        }
        return ans;
    };
    function<vector<int>(int, int)> get_mirror_places = [&] (int graph_node_id, int dir) {
        assert(graph_node_id<graph_x.back().size());
        int position_cur = graph_x.back()[graph_node_id];
        assert(congestionRegionID[dir][graph_node_id]>=0);
        int congestion_region = congestionRegionID[dir][graph_node_id];
        int curx = position_cur/ Y % X, cury = position_cur % Y;
        int trunk_len = congestionRanges[dir][congestion_region].second - congestionRanges[dir][congestion_region].first;
        int max_displace = ratio*float(trunk_len);
        int origional_pos = dir?curx:cury;
        int origional_displacement = calc_displace(origional_pos, dir, congestion_region);
        int init_low = origional_pos;
        int init_high = origional_pos;
        int bound = dir?X:Y;
        while (init_low - 1 >= 0 && calc_displace(init_low - 1, dir, congestion_region) - origional_displacement <= max_displace) init_low--;
        while (init_high + 1 < bound && calc_displace(init_high - 1, dir, congestion_region) - origional_displacement <= max_displace) init_high++;
        int step = 1;
        while ((origional_pos - init_low) / (step + 1) + (init_high - origional_pos) / (step + 1) >= num_tracks) step++;
        init_low = origional_pos - (origional_pos - init_low) / step * step;
        init_high = origional_pos + (init_high - origional_pos) / step * step;
        vector<int> shifts;
        for (double pos = init_low; pos <= init_high; pos += step) {
            int shiftAmount = (pos - origional_pos); 
            if(shiftAmount==0) continue;
            shifts.push_back(pos);
            int min_trunk = congestionRanges[dir][congestion_region].first;
            int max_trunk = congestionRanges[dir][congestion_region].second;
        }
        std::vector<int> indices(shifts.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::vector<int> new_shifts;
        for (int index : indices) {
            new_shifts.push_back(shifts[index]);
        }
    
        shifts = new_shifts;
        return shifts;
    };

    function<void(int, int, int, int, int, vector<vector<int>>, vector<vector<int>>)> dfs_detours = [&] (int x, int par, int par_node_idx, int child_idx, int depth, vector<vector<int>> mirrors, vector<vector<int>> old_mirror_places) {
        if(mirrors.size()==0)
        {
            mirrors.resize(2);
        }
        int size = graph_x.back().size() - 1;
        int position_cur = graph_x.back()[x];
        int curl = position_cur / Y /X, curx = position_cur/ Y % X, cury = position_cur % Y;
        int node_idx = -1;
        if(x==select_root)
        {
            node_idx = create_node(curl,curx,cury, graph_x[x].size());
            par_num_sum_cpu[node_idx+1] += par_num_cpu[node_idx];
            par_num_sum_cpu[node_idx+1] += par_num_sum_cpu[node_idx];
        }
        vector<vector<int>> new_mirrors;
        vector<vector<int>> mirror_places;
        new_mirrors.resize(2);
        mirror_places.resize(2);
        if(old_mirror_places.size()==2)
        for(int dir=0; dir<2; dir++)
        {
            int region_id = getRegionID(congestionRegionID[dir][x], dir);
            int par_region_id = getRegionID(congestionRegionID[dir][par], dir);
            if(region_id >= 0)
            {
                assert(region_id<graph_x.back().size());
                if(getRegionID(congestionRegionID[dir][par], dir)==par)
                {
                    depth+=2;
                }
                if(x==region_id)
                {
                    mirror_places[dir] = get_mirror_places(x, dir);
                }else{
                    mirror_places[dir] = old_mirror_places[dir];
                }
                if(region_id!=x&&mirror_places[dir].size()!=mirrors[dir].size())
                {
                    assert(0);
                }
                assert(region_id==x||mirror_places[dir].size()==mirrors[dir].size());
                for(int m_i = 0; m_i < mirror_places[dir].size(); m_i++)
                {
                    int new_x = dir?mirror_places[dir][m_i]:curx;
                    int new_y = dir?cury:mirror_places[dir][m_i];
                    int new_mirror=-1;
                    if(region_id!=x)
                    {
                        int pre_idx = mirrors[dir][m_i];
                        new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                        assert( mirrors[dir].size()==mirror_places[dir].size());
                        connect_node(pre_idx, new_mirror, child_idx);
                    }
                    else{
                        if(x!=select_root)
                        {
                            int position_par = nodes_cpu[par_node_idx];
                            int parx = position_par/ Y % X, pary = position_par % Y;
                            if(new_x!=parx&&new_y!=pary)
                            {
                                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                                    int midx = pathIndex ? new_x : parx;
                                    int midy = pathIndex ? pary : new_y;
                                    int node_insert = create_node(MAX_LAYER-1, midx, midy, 1);
                                    connect_node(par_node_idx, node_insert, child_idx);
                                    par_num_sum_cpu[node_insert + 1] += par_num_cpu[node_insert];
                                    par_num_sum_cpu[node_insert + 1] += par_num_sum_cpu[node_insert];
                                }
                                new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                                connect_node(new_mirror-1, new_mirror, 0);
                                connect_node(new_mirror-2, new_mirror, 0);
                            }
                            else{
                                new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                                connect_node(par_node_idx, new_mirror, child_idx);
                            }
                        }else{
                            new_mirror = create_node(MAX_LAYER-1, new_x, new_y, graph_x[x].size() - 1);
                        } 
                    }
                    assert(new_mirror>0);
                    new_mirrors[dir].push_back(new_mirror);
                    par_num_sum_cpu[new_mirror + 1] += par_num_cpu[new_mirror];
                    par_num_sum_cpu[new_mirror + 1] += par_num_sum_cpu[new_mirror];
                }
            }
        }
        if(par_node_idx == -1){}
        else {
            int px = nodes_cpu[par_node_idx] / Y % X, py = nodes_cpu[par_node_idx] % Y;
            vector<int> pre_node_idxs;
            vector<int> pre_node_idxs_direct;
            if(px != curx && py != cury)
            {
                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                    int midx = pathIndex ? curx : px;
                    int midy = pathIndex ? py : cury;
                    int pre_node = create_node(MAX_LAYER-1, midx, midy, 1);
                    connect_node(par_node_idx, pre_node, child_idx);
                    par_num_sum_cpu[pre_node + 1] += par_num_cpu[pre_node];
                    par_num_sum_cpu[pre_node + 1] += par_num_sum_cpu[pre_node];                    
                    pre_node_idxs.push_back(pre_node);
                }
                
                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                    int length_edge = pathIndex ? (max(py, cury) - min(py, cury)) : (max(px, curx) - min(px, curx));
                    int max_z_shape = min(10, length_edge);
                    for(int dispace_id = 1; dispace_id < max_z_shape; dispace_id++)
                    {
                        int midx1 = pathIndex ? px : (px*dispace_id+curx*(max_z_shape-dispace_id))/max_z_shape;
                        int midy1 = pathIndex ? (py*dispace_id+cury*(max_z_shape-dispace_id))/max_z_shape : py;
                        if(midx1<0||midy1<0||midx1>=X||midy1>=Y) continue;
                        assert(midx1>=0);
                        assert(midy1>=0);
                        if(midx1==px&&midy1==py) continue;
                        if(midx1==curx&&midy1==cury) continue;
    
                        int midx2 = pathIndex ? curx : (px*dispace_id+curx*(max_z_shape-dispace_id))/max_z_shape;
                        int midy2 = pathIndex ? (py*dispace_id+cury*(max_z_shape-dispace_id))/max_z_shape : cury;
                        if(midx2<0||midy2<0||midx2>=X||midy2>=Y) continue;
                        assert(midx2>=0);
                        assert(midy2>=0);
                        if(midx2==curx&&midy2==cury) continue;
                        if(midx2==px&&midy2==py) continue;
                        int pre_node1 = create_node(MAX_LAYER-1, midx1, midy1, 1);
                        connect_node(par_node_idx, pre_node1, child_idx);
                        par_num_sum_cpu[pre_node1 + 1] += par_num_cpu[pre_node1];
                        par_num_sum_cpu[pre_node1 + 1] += par_num_sum_cpu[pre_node1];
    
                        int pre_node2 = create_node(MAX_LAYER-1, midx2, midy2, 1);
                        connect_node(pre_node1, pre_node2, 0);
                        par_num_sum_cpu[pre_node2 + 1] += par_num_cpu[pre_node2];
                        par_num_sum_cpu[pre_node2 + 1] += par_num_sum_cpu[pre_node2];                   
                        pre_node_idxs.push_back(pre_node2);
                    }
                }
            }
            for(int dir = 0; dir<2; dir++)
            {
                int region_id = getRegionID(congestionRegionID[dir][x], dir);
                int par_region_id = getRegionID(congestionRegionID[dir][par], dir);
                if(par_region_id>=0&&region_id!=par_region_id)
                {
                    for(auto node_par_mirror: mirrors[dir])
                    {
                        int position_par = nodes_cpu[node_par_mirror];
                        int parx = position_par/ Y % X, pary = position_par % Y;
                        if(parx!=curx&&pary!=cury)
                        {
                            for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                                int midx = pathIndex ? curx : parx;
                                int midy = pathIndex ? pary : cury;
                                int node_insert = create_node(MAX_LAYER-1, midx, midy, 1);
                                connect_node(node_par_mirror, node_insert, child_idx);
                                par_num_sum_cpu[node_insert + 1] += par_num_cpu[node_insert] + par_num_sum_cpu[node_insert];
                                pre_node_idxs.push_back(node_insert);
                            }
                        }else{
                            pre_node_idxs_direct.push_back(node_par_mirror);
                        }
                    }
                }
            }
            int connect_parent = par_node_idx;
            int connect_child_idx = child_idx;

            node_idx = create_node(curl, curx, cury, graph_x[x].size() - 1);
            
            if(px == curx || py == cury){
                connect_node(connect_parent, node_idx, connect_child_idx);
            }
            if(x!=select_root)
            {
                for(auto pre_node_idx: pre_node_idxs)
                {
                    connect_node(pre_node_idx, node_idx, 0);
                }
                for(auto pre_node_idx: pre_node_idxs_direct)
                {
                    connect_node(pre_node_idx, node_idx, child_idx);
                }
            }
        }
        depth_max = max(depth_max, node_depth_cpu[node_idx]+1);
        if(x!=select_root)
        {
            par_num_sum_cpu[node_idx+1] += (par_num_sum_cpu[node_idx] + par_num_cpu[node_idx]);
        }
        for(int dir=0; dir<2; dir++)
        {
            int region_id = getRegionID(x, dir);
            int par_region_id = getRegionID(par, dir);
            if(region_id<0) continue;
            if(true)
            {
                int pos2 = dir?cury:curx;
                assert(region_id>=0||region_id<graph_x.back().size());
                int is_tail = congestionRanges[dir][region_id].second==pos2 || congestionRanges[dir][region_id].first==pos2;
                int is_pin = curl < MAX_LAYER - 1;
                assert(x!=select_root);
                if(x!=select_root&&is_pin)
                {
                    for(auto mirror_id: new_mirrors[dir])
                    {
                        int node_duplicate = create_node(curl, curx, cury, 0);
                        child_num_cpu[mirror_id] = graph_x[x].size();// connect in advance
                        connect_node(mirror_id, node_duplicate, graph_x[x].size()-1);
                        par_num_sum_cpu[node_duplicate + 1] += par_num_cpu[node_duplicate];
                        par_num_sum_cpu[node_duplicate + 1] += par_num_sum_cpu[node_duplicate];
                    }
                }
            }
        }
        int idx = 0;
        for(auto e : graph_x[x]) if(e != par)
        {
            depth_max = max(depth_max, depth+1);
            dfs_detours(e, x, node_idx, idx++, depth+1, new_mirrors, mirror_places);
        }
    };
    if(construct_segments)
    {
        rsmt_h_segments.clear();
        rsmt_v_segments.clear();
        rsmt_h_segments.reserve(rsmt.size()*(num_tracks*2));
        rsmt_v_segments.reserve(rsmt.size()*(num_tracks*2));
    }
    points.clear();
    points.reserve(rsmt.size()*(num_tracks*4));
    dfs_detours(select_root, -1, -1, -1, 0, {}, {});
    std::sort(points.begin(), points.end());
    auto last = std::unique(points.begin(), points.end());
    points.erase(last, points.end());
}

void net::generate_detours_reconstruct(bool* const & congestionView_cpu, 
                          float* const & congestionView_xsum_cpu, 
                          float* const & congestionView_ysum_cpu,
                          bool construct_segments, bool display, int net_id){
    

    auto graph_x = rsmt;
    int node_estimate = (graph_x.size()-1)*15;
    int num_edges = 0;

    par_num_cpu.clear();
    par_num_sum_cpu.clear();
    currentChildIDX_cpu.clear();
    par_nodes_cpu.clear();
    child_num_cpu.clear();
    node_depth_cpu.clear();
    nodes_cpu.clear();
    node_index_cnt = 0;

    par_num_cpu.reserve(node_estimate);
    par_num_sum_cpu.reserve(node_estimate);
    currentChildIDX_cpu.reserve(node_estimate);
    par_nodes_cpu.reserve(node_estimate);
    child_num_cpu.reserve(node_estimate);
    node_depth_cpu.reserve(node_estimate);
    nodes_cpu.reserve(node_estimate);

    vector<vector<int> > parents;
    float ratio = 0.15;
    int num_tracks = 9;
    if(mode==0||mode==1)
    {
        ratio = 0.1;
        num_tracks = 6;
    }
    parents.reserve(node_estimate);
    parents.resize(node_estimate);           
    for(auto& row : parents) {
        row.reserve(10);           
    }
    vector<int> child_num_pre(node_estimate, 0);
    int depth_max = 0;
    select_root = select_root_net(graph_x);//.back().size()/2;
    vector<int> congestionRegionID[2];
    vector<vector<pair<int, int> >> congestionRanges;
    congestionRanges.resize(2);
    vector<vector<vector<int>>> stems;
    stems.resize(2);
    congestionRegionID[0].resize(graph_x.size());
    congestionRegionID[1].resize(graph_x.size());
    for(int g_i = 0; g_i < graph_x.size(); g_i++)
    {
        congestionRegionID[0][g_i] = -1;
        congestionRegionID[1][g_i] = -1;
    }
    congestionRanges[0].resize(graph_x.size());
    congestionRanges[1].resize(graph_x.size());
    stems[0].resize(graph_x.size());
    stems[1].resize(graph_x.size());
    function<int(int, int)> getRegionID = [&] (int x, int direction) {
        if(x==-1) return -1;
        if(congestionRegionID[direction][x] == -1) return -1;
        if(congestionRegionID[direction][x] != x)
        {
            if(congestionRegionID[direction][x] == x)
            {
                assert(0);
            }
            int ans = getRegionID(congestionRegionID[direction][x], direction);
            congestionRegionID[direction][x] = ans;
            return ans;
        }else{
            return congestionRegionID[direction][x];
        }
    };
    for(int x = 0; x < graph_x.back().size(); x++)
    {
        int position_cur = graph_x.back()[x];
        int curl = position_cur / Y /X, curx = position_cur / Y % X, cury = position_cur % Y;
        for(int dir=0; dir<2; dir++)
        {
            if(curl<MAX_LAYER-1)
            {
                int stem_pos = dir?curx:cury;
                stems[dir][x].push_back(stem_pos);
            }
            int trunk_pos = dir?cury:curx;
            congestionRanges[dir][x] = make_pair(trunk_pos, trunk_pos);
        }
     
    }
    function<void(int, int)> markCongestion = [&] (int x, int par) {
        int position_cur = graph_x.back()[x];
        int curx = position_cur / Y % X, cury = position_cur % Y;
        for(auto e : graph_x[x]) if(e != par)
        {
            int ex = graph_x.back()[e] / Y % X;
            int ey = graph_x.back()[e] % Y;
            int dir=-1;
            int congestion = -1;
            if(ex == curx)
            {
                bool is_congestion_y = (congestionView_ysum_cpu[curx*Y+max(ey, cury)]-congestionView_ysum_cpu[curx*Y+min(ey, cury)])>0;
                if(is_congestion_y)
                {
                    dir = 1;
                    congestion = 1;
                }
            }
            else if(ey == cury)
            {
                bool is_congestion_x = (congestionView_xsum_cpu[max(ex, curx)*Y+cury]-congestionView_xsum_cpu[min(ex, curx)*Y+cury])>0;
                if(is_congestion_x)
                {
                    dir = 0;
                    congestion = 1;
                }
            }
            else{
                bool is_congestion_y = (congestionView_ysum_cpu[curx*Y+max(ey, cury)]-congestionView_ysum_cpu[curx*Y+min(ey, cury)])>0 || 
                                        (congestionView_ysum_cpu[ex*Y+max(ey, cury)]-congestionView_ysum_cpu[ex*Y+min(ey, cury)])>0;
                if(is_congestion_y)
                {
                    dir = 1;
                    congestion = 2;
                }
                if(congestion==-1)
                {

                    bool is_congestion_x = (congestionView_xsum_cpu[max(ex, curx)*Y+cury]-congestionView_xsum_cpu[min(ex, curx)*Y+cury])>0 || 
                                        (congestionView_xsum_cpu[max(ex, curx)*Y+ey]-congestionView_xsum_cpu[min(ex, curx)*Y+ey])>0;
                    if(is_congestion_x)
                    {
                        dir = 0;
                        congestion = 2;
                    }
                }
            }
            if(congestion == 1)
            {
                int target_region = -1;
                if(x!=select_root)
                {
                    if(congestionRegionID[dir][x]==-1)
                    {
                        congestionRegionID[dir][x] = x;
                    }
                    int region_x = getRegionID(x, dir);
                    target_region = region_x;
                }else{
                    if(congestionRegionID[dir][e]==-1)
                    {
                        congestionRegionID[dir][e] = e;
                    }
                    int region_e = getRegionID(e, dir);
                    target_region = region_e;
                }
                
                congestionRegionID[dir][e] = target_region;
                if(x!=select_root)
                for(auto pos: stems[dir][e])
                {
                    stems[dir][target_region].push_back(pos);
                }
                congestionRanges[dir][target_region].first = min(congestionRanges[dir][target_region].first, congestionRanges[dir][e].first);
                congestionRanges[dir][target_region].second = max(congestionRanges[dir][target_region].second, congestionRanges[dir][e].second);
            }
            else if(congestion == 2)
            {
                if(x!=select_root&&congestionRegionID[dir][x]==-1)//how to merge this? ask its parent?
                {
                    congestionRegionID[dir][x] = x;
                    
                }
                if(congestionRegionID[dir][e]==-1)
                {
                    congestionRegionID[dir][e] = e;
                }
            }
            
            markCongestion(e, x);
        }
        for(int dir=0; dir<2; dir++)
        {
            int x_region = getRegionID(x, dir);
            for(auto e : graph_x[x]) if(e != par)
            {
                int ex = graph_x.back()[e] / Y % X;
                int ey = graph_x.back()[e] % Y;
                int e_region = getRegionID(e, dir);
                if(x_region!=e_region)
                {
                    if(x_region>=0)
                    {
                        stems[dir][x_region].push_back(dir?ex:ey);//外部stem
                        congestionRanges[dir][x_region].first = min(congestionRanges[dir][x_region].first, dir?ey:ex);
                        congestionRanges[dir][x_region].second = max(congestionRanges[dir][x_region].second, dir?ey:ex);
                    }
                    if(e_region>=0)
                    {
                        stems[dir][e_region].push_back(dir?curx:cury);
                        congestionRanges[dir][e_region].first = min(congestionRanges[dir][e_region].first, dir?cury:curx);
                        congestionRanges[dir][e_region].second = max(congestionRanges[dir][e_region].second, dir?cury:curx);
                    }
                }
            }
        }
    };
    markCongestion(select_root, -1);
    function<int(int, int, int)> create_node = [&] (int l, int x, int y) {
        assert(x>=0);
        assert(y>=0);
        int node_idx_insert = node_index_cnt++;
        if(node_idx_insert >= nodes_cpu.size())
        {
            nodes_cpu.emplace_back(0);
            parents.emplace_back(vector<int>());
            parents.back().reserve(10);
            child_num_pre.emplace_back(0);
        }
        nodes_cpu[node_idx_insert] = l * X * Y + x * Y + y;
        return node_idx_insert;
    };
    function<int(int, int, int)> connect_node_pre = [&] (int par_node_index, int cur_index, int cur_child_id) {
        num_edges++;
        int node_idx_insert = cur_index;
        assert(cur_index<nodes_cpu.size());
        assert(par_node_index<nodes_cpu.size());
        int position_cur = nodes_cpu[cur_index];
        int curx = position_cur/ Y % X, cury = position_cur % Y;
        int position_par = nodes_cpu[par_node_index];
        int parx = position_par/ Y % X, pary = position_par % Y;     
        points.emplace_back(curx*Y+cury);
        points.emplace_back(parx*Y+pary);
        assert(curx==parx||cury==pary);
        assert(cur_index<parents.size());
        parents[cur_index].emplace_back(par_node_index*10 + cur_child_id);
        child_num_pre[par_node_index]++;
        return node_idx_insert;
    };

    function<int(int, int, int)> connect_node = [&] (int par_node_index, int cur_index, int cur_child_id) {
        assert(cur_index<nodes_cpu.size());
        assert(par_node_index<nodes_cpu.size());
        int par_x = nodes_cpu[par_node_index] / Y % X, par_y = nodes_cpu[par_node_index] % Y;
        int cur_x = nodes_cpu[cur_index] / Y % X, cur_y = nodes_cpu[cur_index] % Y;
        if(construct_segments)
        {
            if(cur_x==par_x)
            {
                rsmt_v_segments.emplace_back(make_pair(cur_x*Y+min(cur_y, par_y), cur_x*Y+max(cur_y, par_y)));
            }
            if(cur_y==par_y)
            {
                rsmt_h_segments.emplace_back(make_pair(min(cur_x, par_x)+cur_y*X, max(cur_x, par_x)+cur_y*X));
            }
        } 
        if(abs(par_x-cur_x)+abs(par_y-cur_y) < 50)
        {
            return connect_node_pre(par_node_index, cur_index, cur_child_id);
        }
        else if(abs(par_x-cur_x)+abs(par_y-cur_y) < 150)
        {
            int mid_x = (par_x + cur_x) / 2, mid_y = (par_y + cur_y) / 2;
            assert(mid_x==par_x||mid_y==par_y);
            assert(mid_x==cur_x||mid_y==cur_y);
            int mid_point_idx = create_node(MAX_LAYER-1, mid_x, mid_y);
            connect_node_pre(par_node_index, mid_point_idx, cur_child_id);
            connect_node_pre(mid_point_idx, cur_index, 0);
            return cur_index;
        }
        if(par_y==cur_y)
        {
            int mid_x1 = par_x*2/3 +cur_x/3, mid_y1 = par_y;
            int mid_point_idx1 = create_node(MAX_LAYER-1, mid_x1, mid_y1);
            int mid_x2 = par_x*1/3 +cur_x*2/3, mid_y2 = par_y;
            int mid_point_idx2 = create_node(MAX_LAYER-1, mid_x2, mid_y2);
            connect_node_pre(par_node_index, mid_point_idx1, cur_child_id);
            connect_node_pre(mid_point_idx1, mid_point_idx2, 0);
            connect_node_pre(mid_point_idx2, cur_index, 0);
        }
        else if(par_x == cur_x)
        {
            int mid_x1 = par_x, mid_y1 = par_y*2/3 +cur_y/3;
            int mid_point_idx1 = create_node(MAX_LAYER-1, mid_x1, mid_y1);
            int mid_x2 = par_x, mid_y2 = par_y*1/3 +cur_y*2/3;
            int mid_point_idx2 = create_node(MAX_LAYER-1, mid_x2, mid_y2);
            connect_node_pre(par_node_index, mid_point_idx1, cur_child_id);
            connect_node_pre(mid_point_idx1, mid_point_idx2, 0);
            connect_node_pre(mid_point_idx2, cur_index, 0);
        }
        else{
            assert(0);
        }
        return cur_index;
    };

    function<int(int, int, int)> calc_displace = [&] (int query_pos, int dir, int region_id) {
        int ans = 0;
        for(auto pos: stems[dir][region_id])
        {
            ans+=abs(pos-query_pos);
        }
        return ans;
    };
    function<vector<int>(int, int)> get_mirror_places = [&] (int graph_node_id, int dir) {
        assert(graph_node_id<graph_x.back().size());
        int position_cur = graph_x.back()[graph_node_id];
        assert(congestionRegionID[dir][graph_node_id]>=0);
        int congestion_region = congestionRegionID[dir][graph_node_id];
        int curx = position_cur/ Y % X, cury = position_cur % Y;
        int trunk_len = congestionRanges[dir][congestion_region].second - congestionRanges[dir][congestion_region].first;
        int max_displace = ratio*float(trunk_len);
        int origional_pos = dir?curx:cury;
        int origional_displacement = calc_displace(origional_pos, dir, congestion_region);
        int init_low = origional_pos;
        int init_high = origional_pos;
        int bound = dir?X:Y;
        while (init_low - 1 >= 0 && calc_displace(init_low - 1, dir, congestion_region) - origional_displacement <= max_displace) init_low--;
        while (init_high + 1 < bound && calc_displace(init_high - 1, dir, congestion_region) - origional_displacement <= max_displace) init_high++;
        int step = 1;
        while ((origional_pos - init_low) / (step + 1) + (init_high - origional_pos) / (step + 1) >= num_tracks) step++;
        init_low = origional_pos - (origional_pos - init_low) / step * step;
        init_high = origional_pos + (init_high - origional_pos) / step * step;
        vector<int> shifts;
        for (double pos = init_low; pos <= init_high; pos += step) {
            int shiftAmount = (pos - origional_pos); 
            if(shiftAmount==0) continue;
            shifts.push_back(pos);
            int min_trunk = congestionRanges[dir][congestion_region].first;
            int max_trunk = congestionRanges[dir][congestion_region].second;
        }
        std::vector<int> indices(shifts.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }
        std::vector<int> new_shifts;
        for (int index : indices) {
            new_shifts.push_back(shifts[index]);
        }
    
        shifts = new_shifts;
        return shifts;
    };

    function<void(int, int, int, int, int, vector<vector<int>>, vector<vector<int>>)> dfs_detours = [&] (int x, int par, int par_node_idx, int child_idx, int depth, vector<vector<int>> mirrors, vector<vector<int>> old_mirror_places) {
        if(mirrors.size()==0)
        {
            mirrors.resize(2);
        }
        int size = graph_x.back().size() - 1;
        int position_cur = graph_x.back()[x];
        int curl = position_cur / Y /X, curx = position_cur/ Y % X, cury = position_cur % Y;
        int node_idx = -1;
        node_idx = create_node(curl, curx, cury);
        vector<vector<int>> new_mirrors;
        vector<vector<int>> mirror_places;
        new_mirrors.resize(2);
        mirror_places.resize(2);
        if(old_mirror_places.size()==2)
        for(int dir=0; dir<2; dir++)
        {
            int region_id = getRegionID(congestionRegionID[dir][x], dir);
            int par_region_id = getRegionID(congestionRegionID[dir][par], dir);
            if(region_id >= 0)
            {
                assert(region_id<graph_x.back().size());
                if(getRegionID(congestionRegionID[dir][par], dir)==par)
                {
                    depth+=2;
                }
                if(x==region_id)
                {
                    mirror_places[dir] = get_mirror_places(x, dir);
                }else{
                    mirror_places[dir] = old_mirror_places[dir];
                }
                if(region_id!=x&&mirror_places[dir].size()!=mirrors[dir].size())
                {
                    assert(0);
                }
                assert(region_id==x||mirror_places[dir].size()==mirrors[dir].size());
                new_mirrors[dir].reserve(mirror_places[dir].size());
                for(int m_i = 0; m_i < mirror_places[dir].size(); m_i++)
                {
                    int new_x = dir?mirror_places[dir][m_i]:curx;
                    int new_y = dir?cury:mirror_places[dir][m_i];
                    int new_mirror=-1;
                    if(region_id!=x)
                    {
                        int pre_idx = mirrors[dir][m_i];
                        new_mirror = create_node(MAX_LAYER-1, new_x, new_y);
                        assert( mirrors[dir].size()==mirror_places[dir].size());
                        connect_node(pre_idx, new_mirror, child_idx);
                    }
                    else{
                        if(x!=select_root)
                        {
                            int position_par = nodes_cpu[par_node_idx];
                            int parx = position_par/ Y % X, pary = position_par % Y;
                            if(new_x!=parx&&new_y!=pary)
                            {
                                vector<int> nodes_tmp;
                                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                                    int midx = pathIndex ? new_x : parx;
                                    int midy = pathIndex ? pary : new_y;
                                    int node_insert = create_node(MAX_LAYER-1, midx, midy);
                                    connect_node(par_node_idx, node_insert, child_idx);
                                    nodes_tmp.push_back(node_insert);
                                }
                                new_mirror = create_node(MAX_LAYER-1, new_x, new_y);
                                for(auto node_tmp: nodes_tmp)
                                {
                                    connect_node(node_tmp, new_mirror, 0);
                                }
                            }
                            else{
                                new_mirror = create_node(MAX_LAYER-1, new_x, new_y);
                                connect_node(par_node_idx, new_mirror, child_idx);
                            }
                        }else{
                            new_mirror = create_node(MAX_LAYER-1, new_x, new_y);
                        } 
                    }
                    assert(new_mirror>0);
                    new_mirrors[dir].emplace_back(new_mirror);
                }
            }
        }
        if(par_node_idx == -1){}
        else {
            int px = nodes_cpu[par_node_idx] / Y % X, py = nodes_cpu[par_node_idx] % Y;//
            if(px != curx && py != cury)
            {
                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                    int midx = pathIndex ? curx : px;
                    int midy = pathIndex ? py : cury;
                    int pre_node = create_node(MAX_LAYER-1, midx, midy);
                    connect_node(par_node_idx, pre_node, child_idx);
                    connect_node(pre_node, node_idx, 0);
                    
                }
                
                for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                    int length_edge = pathIndex ? (max(py, cury) - min(py, cury)) : (max(px, curx) - min(px, curx));
                    int z_shapes = 10, select = 10;
                    if(mode==0||mode==1) z_shapes = 7, select = 2;
                    int max_z_shape = min(z_shapes, length_edge);
                    pair<float, int> cost_road[10];
                    for(int dispace_id = 1; dispace_id < max_z_shape; dispace_id++)
                    {
                        int midx1 = pathIndex ? px : (px*dispace_id+curx*(max_z_shape-dispace_id))/max_z_shape;
                        int midy1 = pathIndex ? (py*dispace_id+cury*(max_z_shape-dispace_id))/max_z_shape : py;
                        if(midx1<0||midy1<0||midx1>=X||midy1>=Y) continue;
                        assert(midx1>=0);
                        assert(midy1>=0);
                        if(midx1==px&&midy1==py) continue;
                        if(midx1==curx&&midy1==cury) continue;
    
                        int midx2 = pathIndex ? curx : (px*dispace_id+curx*(max_z_shape-dispace_id))/max_z_shape;
                        int midy2 = pathIndex ? (py*dispace_id+cury*(max_z_shape-dispace_id))/max_z_shape : cury;
                        if(midx2<0||midy2<0||midx2>=X||midy2>=Y) continue;
                        assert(midx2>=0);
                        assert(midy2>=0);
                        if(midx2==curx&&midy2==cury) continue;
                        if(midx2==px&&midy2==py) continue;
                        int overflow_num = 0;
                        overflow_num += (congestionView_ysum_cpu[max(midx1, midx2)*Y+max(midy1, midy2)]-congestionView_ysum_cpu[min(midx1, midx2)*Y+min(midy1, midy2)]);
                        overflow_num += (congestionView_ysum_cpu[max(midx1, px)*Y+max(midy1, py)]-congestionView_ysum_cpu[min(midx1, px)*Y+min(midy1, py)]);
                        overflow_num += (congestionView_ysum_cpu[max(curx, midx2)*Y+max(cury, midy2)]-congestionView_ysum_cpu[min(curx, midx2)*Y+min(cury, midy2)]);
                        cost_road[dispace_id-1] = make_pair(overflow_num, dispace_id);

                    }
                    sort(cost_road, cost_road+max_z_shape-1);
                    for(int ii=0;ii<select; ii++)
                    {
                        int dispace_id = cost_road[ii].second;
                        int midx1 = pathIndex ? px : (px*dispace_id+curx*(max_z_shape-dispace_id))/max_z_shape;
                        int midy1 = pathIndex ? (py*dispace_id+cury*(max_z_shape-dispace_id))/max_z_shape : py;
                        if(midx1<0||midy1<0||midx1>=X||midy1>=Y) continue;
                        assert(midx1>=0);
                        assert(midy1>=0);
                        if(midx1==px&&midy1==py) continue;
                        if(midx1==curx&&midy1==cury) continue;
    
                        int midx2 = pathIndex ? curx : (px*dispace_id+curx*(max_z_shape-dispace_id))/max_z_shape;
                        int midy2 = pathIndex ? (py*dispace_id+cury*(max_z_shape-dispace_id))/max_z_shape : cury;
                        if(midx2<0||midy2<0||midx2>=X||midy2>=Y) continue;
                        assert(midx2>=0);
                        assert(midy2>=0);
                        if(midx2==curx&&midy2==cury) continue;
                        if(midx2==px&&midy2==py) continue;
                        int pre_node1 = create_node(MAX_LAYER-1, midx1, midy1);
                        connect_node(par_node_idx, pre_node1, child_idx);
                        points.emplace_back((midx1+midx2)/2*Y+(midy1+midy2)/2);
                        int pre_node2 = create_node(MAX_LAYER-1, midx2, midy2);
                        connect_node(pre_node1, pre_node2, 0);
                        connect_node(pre_node2, node_idx, 0);
                    }
                }
            }
            for(int dir = 0; dir<2; dir++)
            {
                int region_id = getRegionID(congestionRegionID[dir][x], dir);
                int par_region_id = getRegionID(congestionRegionID[dir][par], dir);
                if(par_region_id>=0&&region_id!=par_region_id)
                {
                    for(auto node_par_mirror: mirrors[dir])
                    {
                        int position_par = nodes_cpu[node_par_mirror];
                        int parx = position_par/ Y % X, pary = position_par % Y;
                        if(parx!=curx&&pary!=cury)
                        {
                            for (int pathIndex = 0; pathIndex <= 1; pathIndex++) {
                                int midx = pathIndex ? curx : parx;
                                int midy = pathIndex ? pary : cury;
                                int node_insert = create_node(MAX_LAYER-1, midx, midy);
                                connect_node(node_par_mirror, node_insert, child_idx);
                                connect_node(node_insert, node_idx, 0);
                            }
                        }else{
                            connect_node(node_par_mirror, node_idx, child_idx);
                        }
                    }
                }
            }
            int connect_parent = par_node_idx;
            int connect_child_idx = child_idx;
            if(px == curx || py == cury){
                connect_node(connect_parent, node_idx, connect_child_idx);
            }
        }
        for(int dir=0; dir<2; dir++)
        {
            int region_id = getRegionID(x, dir);
            int par_region_id = getRegionID(par, dir);
            if(region_id<0) continue;
            if(true)
            {
                int pos2 = dir?cury:curx;
                assert(region_id>=0||region_id<graph_x.back().size());
                int is_tail = congestionRanges[dir][region_id].second==pos2 || congestionRanges[dir][region_id].first==pos2;
                int is_pin = curl < MAX_LAYER - 1;
                assert(x!=select_root);
                if(x!=select_root&&is_pin)
                {
                    for(auto mirror_id: new_mirrors[dir])
                    {
                        int node_duplicate = create_node(curl, curx, cury);
                        connect_node(mirror_id, node_duplicate, graph_x[x].size()-1);
                    }
                }
            }
        }
        int idx = 0;
        for(auto e : graph_x[x]) if(e != par)
        {
            depth_max = max(depth_max, depth+1);
            dfs_detours(e, x, node_idx, idx++, depth+1, new_mirrors, mirror_places);
        }
    };
    if(construct_segments)
    {
        rsmt_h_segments.clear();
        rsmt_v_segments.clear();
        rsmt_h_segments.reserve(rsmt.size()*(num_tracks*2));
        rsmt_v_segments.reserve(rsmt.size()*(num_tracks*2));
    }
    points.clear();
    points.reserve(rsmt.size()*(num_tracks*5));
    dfs_detours(select_root, -1, -1, -1, 0, {}, {});    
    if(construct_segments)
    {
        mergeIntervals(rsmt_h_segments);
        mergeIntervals(rsmt_v_segments);
    }
    
    std::sort(points.begin(), points.end());
    auto last = std::unique(points.begin(), points.end());
    points.erase(last, points.end());
    if(construct_segments)
    {
        for(int ii=0; ii<rsmt_h_segments.size();ii++)
        {
            // min(midx, px)+midy*X
            int xpos = rsmt_h_segments[ii].first%X;
            int ypos = rsmt_h_segments[ii].first/X;
            rsmt_h_segments[ii].first = xpos*Y+ypos;
            xpos = rsmt_h_segments[ii].second%X;
            ypos = rsmt_h_segments[ii].second/X;
            rsmt_h_segments[ii].second = xpos*Y+ypos;
        }
    }
    par_num_cpu.resize(node_index_cnt, 0);
    par_num_sum_cpu.resize(node_index_cnt+1, 0);
    child_num_cpu.resize(node_index_cnt, 0);
    node_depth_cpu.resize(node_index_cnt, 0);

    vector<int> visit(node_index_cnt, 0);
    vector<int> nodes_copy(node_index_cnt, 0);
    vector<int> create_id_to_node_id(node_index_cnt, 0);
    node_depth_cpu = vector<int>(node_index_cnt, 0);

    int node_index_copy = 0;
    stack<int> dfs_stack;
    function<void(int)> determinSequence = [&] (int idx) {
        assert(visit[idx]==child_num_pre[idx]);
        dfs_stack.push(idx);
        for(int content_id = 0; content_id < parents[idx].size(); content_id++)
        {
            int content = parents[idx][content_id];
            int parent = content/10;
            visit[parent]++;
            if(visit[parent]==child_num_pre[parent])
            {
                determinSequence(parent);
            }
        }
    };
    for(int i = 0; i < node_index_cnt; i++) visit[i] = 0;
    for(int i = 0; i < node_index_cnt; i++)
    {
        if(child_num_pre[i] == 0)
        {
            determinSequence(i);
        }
    }
    node_index_copy = 0;
    while(!dfs_stack.empty())
    {
        int idx = dfs_stack.top();
        dfs_stack.pop();
        for(int content_id = 0; content_id < parents[idx].size(); content_id++)
        {
            int content = parents[idx][content_id];
            int parent = content/10;
            int child_idx = content%10;
            int sequence_pa = create_id_to_node_id[parent];
            child_num_cpu[sequence_pa] = max(child_num_cpu[sequence_pa], child_idx+1);
            par_nodes_cpu.emplace_back(sequence_pa);
            currentChildIDX_cpu.emplace_back(child_idx);
            node_depth_cpu[node_index_copy] = max(node_depth_cpu[node_index_copy], node_depth_cpu[sequence_pa]+1);
        }
        create_id_to_node_id[idx] = node_index_copy;
        par_num_cpu[node_index_copy] = parents[idx].size();
        par_num_sum_cpu[node_index_copy+1] = par_num_sum_cpu[node_index_copy] + par_num_cpu[node_index_copy];
        nodes_copy[node_index_copy++] = nodes_cpu[idx];
    }

    for(int i = 0; i < node_index_cnt; i++)
    {
        nodes_cpu[i] = nodes_copy[i];
    }
    assert(node_index_copy==node_index_cnt);
    std::sort(points.begin(), points.end());
    auto last2 = std::unique(points.begin(), points.end());
    points.erase(last2, points.end());
}

__global__ void setArrayValue(int* arr, int size, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = value;
    }
}


void build_cuda_database() {
    const int MAX_LEN_INT = 1700000000, MAX_LEN_DOUBLE = 1700000000;
    static int *temp_int = new int[MAX_LEN_INT];
    static double *temp_double = new double[MAX_LEN_DOUBLE];
    static float *temp_float = new float[MAX_LEN_DOUBLE];
    temp_int[0] = 1;
    temp_double[0] = 1.0;
    temp_float[0] = 1.0;
    
    X = db::X;
    L = db::L - 1;
    Y = db::Y;
    XY = max(X, Y);
    nets.reserve(db::nets.size() * 10);
    tmp_nets.resize(db::nets.size());
    int MAX_PIN_SIZE = 12;
    int net_break_count = 0, max_pin_cnt = 1;

    #pragma omp parallel for num_threads(8)
    for(int db_net_id = 0; db_net_id < db::nets.size(); db_net_id++) {
        auto &db_net = db::nets[db_net_id];
        if(db_net.pins.size() == 1) continue;
        vector<net>& nnets = tmp_nets[db_net_id];
        nnets.reserve(1+db_net.pins.size()/MAX_PIN_SIZE);
        if(db_net.pins.size() <= MAX_PIN_SIZE ) {
            net new_net;
            new_net.pins = db_net.pins;
            new_net.original_net_id = db_net_id;
            for(auto &p : new_net.pins) 
                if(p >= X * Y) p -= X * Y;
            nnets.emplace_back(move(new_net));
            db_net.unfinished_subnet_count = 1;
            continue;
        }
        net tmp_net;
        tmp_net.pins = db_net.pins;
        tmp_net.original_net_id = db_net_id;
        for(auto &p : tmp_net.pins) {
            if(p >= X * Y) p -= X * Y;
        }
        tmp_net.construct_rsmt(true);
        vector<bool> not_steiner(tmp_net.rsmt.back().size(), false);
        for(int i = 0;i < tmp_net.rsmt.back().size(); i++){
            if(tmp_net.rsmt.back()[i] < L * X * Y) not_steiner[i] = true;
        }
        vector<int>  sz(tmp_net.rsmt.back().size(), 1), par(tmp_net.rsmt.back().size());
        vector<tuple<int, int, int>> edges;
        edges.reserve(tmp_net.rsmt.back().size() - 1);
        function<int(int)> find_par = [&] (int x) { return x == par[x] ? x : par[x] = find_par(par[x]); };
        for(int i = 0; i < tmp_net.rsmt.back().size(); i++){
            par[i] = i;
            for(auto j : tmp_net.rsmt[i]){
                if(j < i){
                    int x0 = tmp_net.rsmt.back()[i] / Y % X, y0 = tmp_net.rsmt.back()[i] % Y;
                    int x1 = tmp_net.rsmt.back()[j] / Y % X, y1 = tmp_net.rsmt.back()[j] % Y;
                    edges.emplace_back(make_tuple(j, i, not_steiner[i] + not_steiner[j]));// 1-1: pin-pin, 0-0: steiner-steiner, 1-0: pin-steiner
                }
            }
        }
        sort(edges.begin(), edges.end(), [&] (tuple<int, int, int> l, tuple<int, int, int> r) {
            return get<2> (l) < get<2> (r);
        });
        bool big_steiner = false;
        for(auto e : edges) {
            int u = find_par(get<0> (e)), v = find_par(get<1> (e));
            if(u == v) continue;
            if(sz[u] + sz[v] > MAX_PIN_SIZE){
                if(!(not_steiner[get<0>(e)] || not_steiner[get<1>(e)])){
                    big_steiner = true;
                }
                if(!big_steiner) continue;
            }
            if(sz[u] > sz[v]) swap(u, v);
            par[u] = v;
            sz[v] += sz[u];
        }
        vector<vector<int>> new_pins(tmp_net.rsmt.back().size());
        vector<vector<int>> net_edge(tmp_net.rsmt.back().size());
        for(int i = 0; i < tmp_net.rsmt.back().size(); i++){
            new_pins[find_par(i)].emplace_back(tmp_net.rsmt.back()[i]);
        }
        for(int i = 0;i < edges.size(); i++) {
            auto e = edges[i];
            int u = get<0> (e), v = get<1> (e);
            int par_u = find_par(u), par_v = find_par(v);
            if(par_u == par_v)
            {
                net_edge[par_u].emplace_back(i);
                continue;
            }
            if(sz[u] > sz[v])
            {
                swap(u, v);
                swap(par_u, par_v);
            }
            if(!not_steiner[v] && not_steiner[u]){
                swap(u, v);
                swap(par_u, par_v);
            }
            sz[par_u]++;
            if(sz[par_v] == 1)
                par[par_v] = par_u;
            net_edge[par_u].emplace_back(i);
            new_pins[par_u].emplace_back(tmp_net.rsmt.back()[v]);
        }
        for(int i = 0; i < tmp_net.rsmt.back().size(); i++) if(find_par(i) == i) {
            nnets.emplace_back(net());
            nnets.back().rsmt.resize(new_pins[i].size() + 1);
            robin_hood::unordered_map<int,int> indx;
            for(int j = 0;j < new_pins[i].size(); j++){
                if(new_pins[i][j] < L * X * Y)
                    nnets.back().pins.emplace_back(new_pins[i][j]);  //= move(new_pins[i]);
                indx[new_pins[i][j]] = j;

            }
            nnets.back().rsmt.back() = move(new_pins[i]);
            for(auto& j : net_edge[i]){
                auto& e = edges[j];
                int u = get<0> (e), v = get<1> (e);
                if(find_par(v) == i){
                    swap(u, v);
                }
                int x = indx[tmp_net.rsmt.back()[u]];
                int y = indx[tmp_net.rsmt.back()[v]];
                nnets.back().rsmt[x].emplace_back(y);
                nnets.back().rsmt[y].emplace_back(x);
            }
            
            nnets.back().original_net_id = db_net_id;
        }
        // exit(0);
        db_net.unfinished_subnet_count = nnets.size();
    }
    for(int db_net_id = 0; db_net_id < db::nets.size(); db_net_id++) {
        auto &db_net = db::nets[db_net_id];
        net_break_count += tmp_nets[db_net_id].size() != 1;
        for(auto &nnet : tmp_nets[db_net_id]){
            db_net.subnets.emplace_back(nets.size());
            max_pin_cnt = max(max_pin_cnt, (int)nnet.pins.size());
            nets.emplace_back(move(nnet));
        }
    }
    pin_cnt_sum_cpu.reserve(1 + nets.size());
    pin_cnt_sum_cpu.resize(1 + nets.size(), 0);
    pin_cnt_sum_phase2_cpu.reserve(1 + nets.size());
    pin_cnt_sum_phase2_cpu.resize(1 + nets.size(), 0);
    ripup_flag_cpu.reserve(1 + nets.size());
    ripup_flag_cpu.resize(1 + nets.size(), 0);
    #pragma omp parallel for num_threads(8)
    for(int i = 0; i < nets.size(); i++) {
        nets[i].calc_hpwl();
    }
    for(int i = 0; i < nets.size(); i++) {
        pin_cnt_sum_cpu[i + 1] = pin_cnt_sum_cpu[i] + nets[i].pins.size();
    }
    printf("\n    MAX PINS: %d\n", max_pin_cnt);
    printf("    THRESH PINS#: %d\n", MAX_PIN_SIZE);
    printf("    Broken Nets: %d\n", net_break_count);
    
    NET_NUM = nets.size();


    DIR = db::layers[1].dir;
    unit_length_wire_cost = db::unit_length_wire_cost;
    unit_via_cost = db::unit_via_cost;

    assert(X - 1 <= MAX_LEN_INT);
    for(int i = 0; i < X - 1; i++) temp_int[i] = db::x_edge_len[i];
    cudaMalloc(&x_edge_len, (X - 1) * sizeof(int));
    cudaMemcpy(x_edge_len, temp_int, (X - 1) * sizeof(int), cudaMemcpyHostToDevice);


    assert(Y - 1 <= MAX_LEN_INT);
    for(int i = 0; i < Y - 1; i++) temp_int[i] = db::y_edge_len[i];
    cudaMalloc(&y_edge_len, (Y - 1) * sizeof(int));
    cudaMemcpy(y_edge_len, temp_int, (Y - 1) * sizeof(int), cudaMemcpyHostToDevice);

    assert(L <= MAX_LEN_DOUBLE);
    for(int i = 0; i < L; i++) temp_double[i] = db::unit_length_short_costs[i + 1];
    cudaMalloc(&unit_length_short_costs, L * sizeof(double));
    cudaMemcpy(unit_length_short_costs, temp_double, L * sizeof(double), cudaMemcpyHostToDevice);

    assert(L <= MAX_LEN_DOUBLE);
    for(int i = 0; i < L; i++) temp_double[i] = db::layers[i + 1].min_len;
    cudaMalloc(&layer_min_len, L * sizeof(double));
    cudaMemcpy(layer_min_len, temp_double, L * sizeof(double), cudaMemcpyHostToDevice);


    assert(L * X * Y <= MAX_LEN_DOUBLE);
    for(int l = 0; l < L; l++)
        for(int x = 0; x < X; x++)
            for(int y = 0; y < Y; y++)
                temp_float[IDX(l, x, y)] = db::capacity[l + 1][x][y];
    cudaMalloc(&capacity, L * X * Y * sizeof(float));
    cudaMemcpy(capacity, temp_float, L * X * Y * sizeof(float), cudaMemcpyHostToDevice);
    

    cudaMalloc(&pin_acc_num, (1 + NET_NUM) * sizeof(int));
    cudaMalloc(&pin_acc_num_phase2, (1 + NET_NUM) * sizeof(int));
    setArrayValue<<<BLOCK_NUM((1 + NET_NUM)), THREAD_NUM>>>(pin_acc_num, 1 + NET_NUM, -1);
    cudaMemcpy(pin_acc_num, pin_cnt_sum_cpu.data(), (1 + NET_NUM) * sizeof(int), cudaMemcpyHostToDevice);
    PIN_NUM = pin_cnt_sum_cpu.back();

    cudaMalloc(&ripup_flag, (1 + NET_NUM) * sizeof(int));
    cudaMemset(ripup_flag, 0, (NET_NUM+1) * sizeof(int));

    if(LOG) cerr << "PIN_NUM " << PIN_NUM << endl;

    
    assert(PIN_NUM <= MAX_LEN_INT);

    for(auto &dbnet : db::nets)
        for(auto pin : dbnet.pins) 
            if(pin < X * Y) total_via_count++;

    for(int i = 0, pin_id = 0; i < NET_NUM; i++)
        for(auto pin : nets[i].pins) temp_int[pin_id++] = pin;
    cudaMalloc(&pins, PIN_NUM * sizeof(int));
    cudaMemcpy(pins, temp_int, PIN_NUM * sizeof(int), cudaMemcpyHostToDevice);

    net_x_cpu.resize(NET_NUM);
    net_y_cpu.resize(NET_NUM);
    

    all_track_cnt = 0;
    for(int l = 0; l < L; l++) all_track_cnt += (l & 1 ^ DIR) ? X : Y;
    assert(all_track_cnt <= MAX_LEN_INT);
    for(int l = 0, cnt = 0; l < L; l++) {
        if((l & 1 ^ DIR) == 0) for(int y = 0; y < Y; y++) temp_int[cnt++] = l * XY + y;
        if((l & 1 ^ DIR) == 1) for(int x = 0; x < X; x++) temp_int[cnt++] = l * XY + x;
        if(l + 1 == L) assert(cnt == all_track_cnt);
    }
    cudaMalloc(&idx2track, all_track_cnt * sizeof(int));
    cudaMemcpy(idx2track, temp_int, sizeof(int) * all_track_cnt, cudaMemcpyHostToDevice);


    cudaMalloc(&congestion, X * Y * sizeof(bool));
    cudaMalloc(&congestion_xsum, X * Y * sizeof(float));
    cudaMalloc(&congestion_ysum, X * Y * sizeof(float));

    cudaMallocManaged(&routes, (ROUTE_PER_PIN * PIN_NUM+1) * sizeof(int));

    cudaMalloc(&wcost, L * X * Y * sizeof(float));
    cudaMalloc(&vcost, L * X * Y * sizeof(float));
    cudaMalloc(&cross_points, X * Y * sizeof(int));
    cudaMemset(cross_points, 0, X * Y * sizeof(bool));
    cudaMalloc(&presum, L * X * Y * sizeof(double));
    cudaMalloc(&demand, L * X * Y * sizeof(float));
    cudaMalloc(&pre_demand, L * X * Y * sizeof(int));



    cudaMalloc(&net_ids, NET_NUM * sizeof(int));
    cudaMalloc(&is_of_net, NET_NUM * sizeof(bool));
    cudaMalloc(&of_edge_sum, L * X * Y * sizeof(int));
    cudaMalloc(&last, L * X * Y * sizeof(int));
    cudaMalloc(&timestamp, L * X * Y * sizeof(int));


    

    cudaMemset(demand, 0, sizeof(float) * L * X * Y);
    cudaMemset(timestamp, 0, sizeof(int) * L * X * Y);
    cudaMemset(pre_demand, 0, sizeof(int) * L * X * Y);

    net_ids_cpu = new int[NET_NUM];
}

}

using namespace cudb;

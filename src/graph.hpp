
#pragma once
#include "global.h"
#include <fcntl.h>
#include <unistd.h>
#include "database_cuda.hpp"



namespace graph {

//declaration
void update_cost();
void output(char file_name[]);
__global__ void compute_presum();

//implementation

__device__ double wire_segment_cost(int layer, int xmin, int xmax, int ymin, int ymax) {
    return presum[IDX(layer, xmax, ymax)] - presum[IDX(layer, xmin, ymin)];
}

__device__ double of_cost_scaled(float capacity, float demand) {
    if(capacity > 0.001) return __expf(min(0.5 * (demand - capacity), of_cost_scale * 0.5 * (demand - capacity)));
    if(demand > 0) return __expf(min(1.5 * demand, of_cost_scale * 1.5 * demand));
    return 0;
}
__device__ double of_cost(float capacity, float demand) {
    if(capacity > 0.001) return __expf(0.5 * (demand - capacity));
    if(demand > 0) return __expf(1.5 * demand);
    return 0;
}

__device__ double incremental_of_cost(int l, int x, int y, double incre_demand, double incre_base = 0) {
    int idx = l * X * Y + x * Y + y;
    return min(1e12, (of_cost_scaled(capacity[idx], demand[idx] + incre_demand + incre_base) 
                    - of_cost_scaled(capacity[idx], demand[idx] + incre_base)) * unit_length_short_costs[l]);
}

__global__ void update_wcost_cuda_ispd24() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y, dir = l & 1 ^ DIR;
    wcost[idx] = incremental_of_cost(l, x, y, 1);
    if(dir == 0 && x < X - 1) wcost[idx] += unit_length_wire_cost * x_edge_len[x];//@@@@
    if(dir == 1 && y < Y - 1) wcost[idx] += unit_length_wire_cost * y_edge_len[y];
}



__device__ const double via_weight = 1;
__device__ void update_single_vcost_ispd24(int l, int x, int y) {
    int idx = IDX(l, x, y);
    int is_cross_point = cross_points[x * Y + y];
    if(l & 1 ^ DIR) {
        if(y == 0) 
            vcost[idx] = incremental_of_cost(l, x, y, 1, is_cross_point);
        else if(y == Y - 1)
            vcost[idx] = incremental_of_cost(l, x, y - 1, 1, is_cross_point);
        else
            vcost[idx] = incremental_of_cost(l, x, y, 0.5, is_cross_point) + incremental_of_cost(l, x, y - 1, 0.5, is_cross_point);
    } else {
        if(x == 0)
            vcost[idx] = incremental_of_cost(l, x, y, 1, is_cross_point);
        else if(x == X - 1)
            vcost[idx] = incremental_of_cost(l, x - 1, y, 1, is_cross_point);
        else
            vcost[idx] = incremental_of_cost(l, x, y, 0.5, is_cross_point) + incremental_of_cost(l, x - 1, y, 0.5, is_cross_point);
    }
    vcost[idx] += unit_via_cost;
}


__global__ void compute_presum_general(int *to_sum) {
    extern __shared__ int sum2[];
    if(threadIdx.x == 0) sum2[0] = 0;
    int l = idx2track[blockIdx.x] / XY;
    if(l & 1 ^ DIR) {
        int x = idx2track[blockIdx.x] % XY;
        for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum2[y + 1] = to_sum[IDX(l, x, y)];
        __syncthreads();
        for(int d = 0; (1 << d) < Y; d++) {
            for(int idx = threadIdx.x; idx < Y; idx += blockDim.x) 
                if(idx >> d & 1) sum2[idx] += sum2[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int y = threadIdx.x; y < Y; y += blockDim.x) to_sum[IDX(l, x, y)] = sum2[y];
    } else {
        int y = idx2track[blockIdx.x] % XY;
        for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum2[x + 1] = to_sum[IDX(l, x, y)];
        __syncthreads();
        for(int d = 0; (1 << d) < X; d++) {
            for(int idx = threadIdx.x; idx < X; idx += blockDim.x) 
                if(idx >> d & 1) sum2[idx] += sum2[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int x = threadIdx.x; x < X; x += blockDim.x) to_sum[IDX(l, x, y)] = sum2[x];
    }
}

__global__ void update_vcost_ispd24() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < L * X * Y) update_single_vcost_ispd24(idx / X / Y, idx / Y % X, idx % Y);
}
__device__ void atomic_add_unit_demand_wire_segment(int l, int minx, int maxx, int miny, int maxy, int stamp, int K = 1) {
    assert(minx == maxx || miny == maxy);

    atomicAdd(pre_demand + IDX(l, minx, miny), K);
    if(minx == maxx) {
        atomicAdd(pre_demand + IDX(l, minx, maxy), -K);
        timestamp[IDX(l, minx, maxy)] = stamp;
    }
    if(miny == maxy) {
        atomicAdd(pre_demand + IDX(l, maxx, miny), -K);
        timestamp[IDX(l, maxx, miny)] = stamp;
    }
}
__global__ void commit_all_edge(int stamp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y;
    if((l & 1 ^ DIR) == 0 && x + 1 < X && pre_demand[IDX(l, x + 1, y)] > 0) {
        demand[IDX(l, x, y)] += pre_demand[IDX(l, x + 1, y)];
        timestamp[IDX(l, x, y)] = stamp;
        atomicAdd(&total_wirelength, pre_demand[IDX(l, x + 1, y)] * x_edge_len[x]);
    }
    if((l & 1 ^ DIR) == 1 && y + 1 < Y && pre_demand[IDX(l, x, y + 1)] > 0) {
        demand[IDX(l, x, y)] += pre_demand[IDX(l, x, y + 1)];
        timestamp[IDX(l, x, y)] = stamp;
        atomicAdd(&total_wirelength, pre_demand[IDX(l, x, y + 1)] * y_edge_len[y]);
    }
}
void batch_wire_update(int stamp) {
    compute_presum_general<<<all_track_cnt, THREAD_NUM, sizeof(int) * XY>>> (pre_demand);
    commit_all_edge<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> (stamp);
    cudaMemset(pre_demand, 0, sizeof(int) * L * X * Y);
}


void update_cost_ispd24() {
    update_wcost_cuda_ispd24<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> ();
    update_vcost_ispd24<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> ();
}
void update_cost() {
    update_cost_ispd24();
}

__global__ void add_all_overflow_cost() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y;
    if(((l & 1 ^ DIR) == 0 && x + 1 < X) || ((l & 1 ^ DIR) == 1 && y + 1 < Y)) {
        double c = unit_length_short_costs[l] * of_cost(capacity[idx], demand[idx]);
        atomicAdd(&total_overflow_cost, c);
        atomicAdd(&layer_overflow_cost[l], c);
    }
}

void report_score() {
    printf("\n----------------------------------------------------------------------\n");
    total_overflow_cost = 0;
    for(int i = 0; i < L; i++) layer_overflow_cost[i] = 0;
    add_all_overflow_cost<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> ();
    cudaDeviceSynchronize();
    printf("             WL            Via                  OF                                             Total\n%15.0f%15.0f%20.0f%50.0f\n", 
        total_wirelength * db::unit_length_wire_cost, total_via_count * db::unit_via_cost, total_overflow_cost, 
        total_wirelength * db::unit_length_wire_cost + total_via_count * db::unit_via_cost + total_overflow_cost);
    printf("----------------------------------------------------------------------\n\n");
}

void finish_nets(vector<int> &finished_nets) {
    for(auto _net_id : finished_nets) {
        int _ = nets[_net_id].original_net_id;
        if(--db::nets[_].unfinished_subnet_count == 0) nets2output.push(_);

    }
}

void output_nets() {
    auto write_int = [&] (int num) {
        if(num == 0) {
            putc_unlocked('0', out_file);
        } else {
            static char temp[20];
            int len = 0;
            while(num) temp[len++] = num % 10, num /= 10;
            for(int i = len - 1; i >= 0; i--) putc_unlocked('0' + temp[i], out_file);
        }
    };
    auto write_single_route = [&] (int x0, int y0, int l0, int x1, int y1, int l1) {
        write_int(x0);
        putc_unlocked(' ', out_file);
        write_int(y0);
        putc_unlocked(' ', out_file);
        write_int(l0);
        putc_unlocked(' ', out_file);
        write_int(x1);
        putc_unlocked(' ', out_file);
        write_int(y1);
        putc_unlocked(' ', out_file);
        write_int(l1);
        putc_unlocked('\n', out_file);
    };
    auto write_route = [&] (int pos0, int pos1) {
        if(pos0 > pos1) swap(pos0, pos1);
        write_single_route(pos0 / Y % X, pos0 % Y, pos0 / X / Y, pos1 / Y % X, pos1 % Y, pos1 / X / Y);
        // write_single_route(pos0 / Y % X, pos0 % (X*Y) / X, pos0 / X / Y, pos1 / Y % X, pos1 % (X*Y) / X, pos1 / X / Y);
    };

    while(!nets2output.empty()) {
        int _ = nets2output.front();
        nets2output.pop();
        for(auto e : db::nets[_].name) putc_unlocked(e, out_file);
        putc_unlocked('\n', out_file);
        putc_unlocked('(', out_file);
        putc_unlocked('\n', out_file);
        for(auto pin : db::nets[_].pins) if(pin < X * Y) write_route(pin, pin + X * Y);
        for(int i = 0; i < db::nets[_].extra_routes.size(); i += 2) 
            write_route(db::nets[_].extra_routes[i], db::nets[_].extra_routes[i + 1]);
        for(auto net_id : db::nets[_].subnets) {
            // int *net_routes = routes + pin_cnt_sum_cpu[net_id] * ROUTE_PER_PIN;
            int *net_routes;
            if(ripup_flag_cpu[net_id]==0)
            {
                net_routes = routes + pin_cnt_sum_cpu[net_id] * ROUTE_PER_PIN;
            }
            else{
                net_routes = routes_phase2 + pin_cnt_sum_phase2_cpu[net_id] * ROUTE_PER_PIN_PHASE2;
            }
            for(int j = 1; j < net_routes[0]; j += 2) write_route(net_routes[j] + X * Y, net_routes[j + 1] + X * Y);
        }
        putc_unlocked(')', out_file);
        putc_unlocked('\n', out_file);
    }
}

void output_putc(char file_name[]) {
    double output_start_time = elapsed_time();

    FILE *file = fopen(file_name, "w");
    std::setbuf(file, output_buffer);

    vector<int> is_pin(X * Y, -1);
    static int *routes_cpu = new int[PIN_NUM * ROUTE_PER_PIN]; 
    static int *routes_phase2_cpu = new int[PIN_NUM_PHASE2 * ROUTE_PER_PIN_PHASE2];
    cudaMemcpy(routes_cpu, routes, sizeof(int) * PIN_NUM * ROUTE_PER_PIN, cudaMemcpyDeviceToHost);
    cudaMemcpy(routes_phase2_cpu, routes_phase2, sizeof(int) * PIN_NUM_PHASE2 * ROUTE_PER_PIN_PHASE2, cudaMemcpyDeviceToHost);

    auto write_int = [&] (int num) {
        if(num == 0) {
            putc_unlocked('0', file);
        } else {
            static char temp[20];
            int len = 0;
            while(num) temp[len++] = num % 10, num /= 10;
            for(int i = len - 1; i >= 0; i--) putc_unlocked('0' + temp[i], file);
        }
    };
    auto write_single_route = [&] (int x0, int y0, int l0, int x1, int y1, int l1) {
        write_int(x0);
        putc_unlocked(' ', file);
        write_int(y0);
        putc_unlocked(' ', file);
        write_int(l0);
        putc_unlocked(' ', file);
        write_int(x1);
        putc_unlocked(' ', file);
        write_int(y1);
        putc_unlocked(' ', file);
        write_int(l1);
        putc_unlocked('\n', file);
    };
    auto write_route = [&] (int pos0, int pos1) {
        if(pos0 > pos1) swap(pos0, pos1);
        write_single_route(pos0 / Y % X, pos0 % Y, pos0 / X / Y, pos1 / Y % X, pos1 % Y, pos1 / X / Y);
    };

    for(int _ = 0; _ < db::nets.size(); _++) {
        for(auto e : db::nets[_].name) putc_unlocked(e, file);
        putc_unlocked('\n', file);
        putc_unlocked('(', file);
        putc_unlocked('\n', file);
        for(auto pin : db::nets[_].pins) if(pin < X * Y) is_pin[pin] = _;
        for(int i = 0; i < db::nets[_].extra_routes.size(); i += 2) 
            write_route(db::nets[_].extra_routes[i], db::nets[_].extra_routes[i + 1]);
        for(auto net_id : db::nets[_].subnets) {
            auto &net = nets[net_id];
            // int *net_routes = routes_cpu + pin_cnt_sum_cpu[net_id] * ROUTE_PER_PIN;
            int *net_routes;
            if(ripup_flag_cpu[net_id]==0)
            {
                net_routes = routes_cpu + pin_cnt_sum_cpu[net_id] * ROUTE_PER_PIN;
            }
            else{
                net_routes = routes_phase2_cpu + pin_cnt_sum_phase2_cpu[net_id] * ROUTE_PER_PIN_PHASE2;
            }
            for(int j = 1; j < net_routes[0]; j += 2) {
                int pos0 = min(net_routes[j], net_routes[j + 1]), pos1 = max(net_routes[j], net_routes[j + 1]);
                int l0 = pos0 / X / Y + 1, x0 = pos0 / Y % X, y0 = pos0 % Y;
                int l1 = pos1 / X / Y + 1, x1 = pos1 / Y % X, y1 = pos1 % Y;
                if(l0 == 1 && l0 < l1 && is_pin[x0 * Y + y0] == _) l0 = 0, is_pin[x0 * Y + y0] = -1;
                write_single_route(x0, y0, l0, x1, y1, l1);
            }
        }
        for(auto pin : db::nets[_].pins) 
            if(pin < X * Y && is_pin[pin] == _) write_route(pin, pin + X * Y);
        putc_unlocked(')', file);
        putc_unlocked('\n', file);
    }
    fclose(file);
    output_time = elapsed_time() - output_start_time;
}



void output(char file_name[]) {
    double output_start_time = elapsed_time();
    FILE *file = fopen(file_name, "w");
    int char_cur = 0;
    const int CHAR_COUNT = 1000000000;
    static char out[CHAR_COUNT];

    int write_count = 0;
    vector<int> is_pin(X * Y, -1);
    static int *routes_cpu = new int[PIN_NUM * ROUTE_PER_PIN];
    static int *routes_phase2_cpu = new int[PIN_NUM_PHASE2 * ROUTE_PER_PIN_PHASE2];
    cudaMemcpy(routes_cpu, routes, sizeof(int) * PIN_NUM * ROUTE_PER_PIN, cudaMemcpyDeviceToHost);
    cudaMemcpy(routes_phase2_cpu, routes_phase2, sizeof(int) * PIN_NUM_PHASE2 * ROUTE_PER_PIN_PHASE2, cudaMemcpyDeviceToHost);

    auto write_int = [&] (int num) {
        if(num == 0)
            out[char_cur++] = '0';
        else {
            static char temp[20];
            int len = 0;
            while(num) temp[len++] = num % 10, num /= 10;
            for(int i = len - 1; i >= 0; i--) out[char_cur++] = temp[i] + '0';
        }
    };
    auto write_single_route = [&] (int x0, int y0, int l0, int x1, int y1, int l1) {
        write_int(x0);
        out[char_cur++] = ' ';
        write_int(y0);
        out[char_cur++] = ' ';
        write_int(l0);
        out[char_cur++] = ' ';
        write_int(x1);
        out[char_cur++] = ' ';
        write_int(y1);
        out[char_cur++] = ' ';
        write_int(l1);
        out[char_cur++] = '\n';
    };
    auto write_route = [&] (int pos0, int pos1) {
        if(pos0 > pos1) swap(pos0, pos1);
        write_single_route(pos0 / Y % X, pos0 % Y, pos0 / X / Y, pos1 / Y % X, pos1 % Y, pos1 / X / Y);
    };

    for(int _ = 0; _ < db::nets.size(); _++) {
        if(char_cur * 1.1 > CHAR_COUNT) {
            fwrite(out, sizeof(char), char_cur, file), char_cur = 0;
            write_count++;
        }
        for(auto e : db::nets[_].name) out[char_cur++] = e;
        out[char_cur++] = '\n';
        out[char_cur++] = '(';
        out[char_cur++] = '\n';
        for(auto pin : db::nets[_].pins) if(pin < X * Y) is_pin[pin] = _;
        for(int i = 0; i < db::nets[_].extra_routes.size(); i += 2) 
            write_route(db::nets[_].extra_routes[i], db::nets[_].extra_routes[i + 1]);
        for(auto net_id : db::nets[_].subnets) {
            auto &net = nets[net_id];
            // int *net_routes = routes_cpu + pin_cnt_sum_cpu[net_id] * ROUTE_PER_PIN;
            int *net_routes;
            if(ripup_flag_cpu[net_id]==0)
            {
                net_routes = routes_cpu + pin_cnt_sum_cpu[net_id] * ROUTE_PER_PIN;
            }
            else{
                net_routes = routes_phase2_cpu + pin_cnt_sum_phase2_cpu[net_id] * ROUTE_PER_PIN_PHASE2;
            }
            for(int j = 1; j < net_routes[0]; j += 2) {
                int pos0 = min(net_routes[j], net_routes[j + 1]), pos1 = max(net_routes[j], net_routes[j + 1]);
                int l0 = pos0 / X / Y + 1, x0 = pos0 / Y % X, y0 = pos0 % Y;
                int l1 = pos1 / X / Y + 1, x1 = pos1 / Y % X, y1 = pos1 % Y;
                if(l0 == 1 && l0 < l1 && is_pin[x0 * Y + y0] == _) l0 = 0, is_pin[x0 * Y + y0] = -1;
                write_single_route(x0, y0, l0, x1, y1, l1);
            }
        }
        for(auto pin : db::nets[_].pins) 
            if(pin < X * Y && is_pin[pin] == _) write_route(pin, pin + X * Y);
        out[char_cur++] = ')';
        out[char_cur++] = '\n';
    }
    
    double fwrite_start_time = elapsed_time();
    fwrite(out, sizeof(char), char_cur, file), char_cur = 0;
    //write(file, out, char_cur);
    fclose(file);
    printf("    write calls: %d\n", ++write_count);
    printf("    fwrite time: %.2f\n", elapsed_time() - fwrite_start_time);
    output_time = elapsed_time() - output_start_time;
}

__global__ void compute_presum() {
    extern __shared__ double sum[];
    if(threadIdx.x == 0) sum[0] = 0;
    int l = idx2track[blockIdx.x] / XY;
    if(l & 1 ^ DIR) {
        int x = idx2track[blockIdx.x] % XY;
        for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum[y + 1] = wcost[l * X * Y + x * Y + y];
        __syncthreads();
        for(int d = 0; (1 << d) < Y; d++) {
            for(int idx = threadIdx.x; idx < Y; idx += blockDim.x) 
                if(idx >> d & 1) sum[idx] += sum[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int y = threadIdx.x; y < Y; y += blockDim.x) presum[l * X * Y + x * Y + y] = sum[y];
    } else {
        int y = idx2track[blockIdx.x] % XY;
        for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum[x + 1] = wcost[l * X * Y + x * Y + y];
        __syncthreads();
        for(int d = 0; (1 << d) < X; d++) {
            for(int idx = threadIdx.x; idx < X; idx += blockDim.x) 
                if(idx >> d & 1) sum[idx] += sum[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int x = threadIdx.x; x < X; x += blockDim.x) presum[l * X * Y + x * Y + y] = sum[x];
    }
}

__global__ void mark_overflow_edges(int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y;
    if(((l & 1 ^ DIR) == 0 && x + 1 < X) || ((l & 1 ^ DIR) == 1 && y + 1 < Y))
        of_edge_sum[idx] = (capacity[idx] + threshold <= demand[idx] ? 1 : 0);
}


__global__ void mark_overflow_nets() {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(net_id >= NET_NUM) return;
    is_of_net[net_id] = false;
    int *net_routes;
    if(ripup_flag[net_id]==0)
    {
        net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    }
    else{
        assert(0);
        net_routes = routes_phase2 + pin_acc_num_phase2[net_id] * ROUTE_PER_PIN_PHASE2;
    }
    // int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    int of_count = 0, pin_cnt = pin_acc_num[net_id + 1] - pin_acc_num[net_id];
    for(int i = 1; i < net_routes[0]; i += 2) {
        int l = net_routes[i] / X / Y, x0 = net_routes[i] / Y % X, y0 = net_routes[i] % Y;
        int x1 = net_routes[i + 1] / Y % X, y1 = net_routes[i + 1] % Y;
        if(x0 != x1) of_count += of_edge_sum[IDX(l, max(x0, x1), y0)] - of_edge_sum[IDX(l, min(x0, x1), y0)];
        if(y0 != y1) of_count += of_edge_sum[IDX(l, x0, max(y0, y1))] - of_edge_sum[IDX(l, x0, min(y0, y1))];
        if(of_count > 10) {
            is_of_net[net_id] = true;
            break;
        }
    }
}

__global__ void commit_wire_demand(int net_cnt, int net_offset, int stamp, int K = 1, int need_work_dir = -1) {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(net_id >= net_cnt) return;
    net_id = net_ids[net_id + net_offset];
    int *net_routes;
    if(ripup_flag[net_id]==0 || need_work_dir==0)
    {
        net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    }
    else{
        // assert(0);
        net_routes = routes_phase2 + pin_acc_num_phase2[net_id] * ROUTE_PER_PIN_PHASE2;
    }
    // int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    double wirelength = 0;
    //commit wires
    for(int i = 1; i < net_routes[0]; i += 2) if(net_routes[i] / X / Y == net_routes[i + 1] / X / Y) {
        int l = net_routes[i] / X / Y;
        int x0 = net_routes[i] / Y % X, y0 = net_routes[i] % Y;
        int x1 = net_routes[i + 1] / Y % X, y1 = net_routes[i + 1] % Y;
        if(x0 < x1) for(int x = x0; x < x1; x++) atomicAdd(demand + IDX(l, x, y0), K), wirelength += x_edge_len[x], timestamp[IDX(l, x, y0)] = stamp;
        if(x1 < x0) for(int x = x1; x < x0; x++) atomicAdd(demand + IDX(l, x, y0), K), wirelength += x_edge_len[x], timestamp[IDX(l, x, y0)] = stamp;
        if(y0 < y1) for(int y = y0; y < y1; y++) atomicAdd(demand + IDX(l, x0, y), K), wirelength += y_edge_len[y], timestamp[IDX(l, x0, y)] = stamp;
        if(y1 < y0) for(int y = y1; y < y0; y++) atomicAdd(demand + IDX(l, x0, y), K), wirelength += y_edge_len[y], timestamp[IDX(l, x0, y)] = stamp;
        timestamp[IDX(l, x0, y0)] = timestamp[IDX(l, x0, y1)] = timestamp[IDX(l, x1, y0)] = timestamp[IDX(l, x1, y1)] = stamp;
    }
    atomicAdd(&total_wirelength, wirelength * K);
}
__global__ void commit_via_demand(int net_cnt, int net_offset, int stamp, int K = 1, int need_work_dir = -1) {
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(net_id >= net_cnt) return;
    net_id = net_ids[net_id + net_offset];
    int *net_routes;
    if(ripup_flag[net_id]==0 || need_work_dir==0)
    {
        net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    }
    else{
        net_routes = routes_phase2 + pin_acc_num_phase2[net_id] * ROUTE_PER_PIN_PHASE2;
    }
    // int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    int via_count = 0;
    //commit vias
    for(int i = 1; i < net_routes[0]; i += 2) if(net_routes[i] / X / Y != net_routes[i + 1] / X / Y) {
        int x = net_routes[i] / Y % X, y = net_routes[i] % Y, l0 = net_routes[i] / X / Y, l1 = net_routes[i + 1] / X / Y;
        int minl = min(l0, l1), maxl = max(l0, l1);
        via_count += maxl - minl;
        for(int l = minl; l < maxl; l++) {
            if(timestamp[IDX(l, x, y)] < stamp) {
                
                if(l & 1 ^ DIR) {
                    if(y == 0) 
                        atomicAdd(demand + IDX(l, x, y), K);
                    else if(y == Y - 1) 
                        atomicAdd(demand + IDX(l, x, y - 1), K);
                    else {
                        atomicAdd(demand + IDX(l, x, y - 1), 0.5 * K);
                        atomicAdd(demand + IDX(l, x, y), 0.5 * K);
                    }
                } else {
                    if(x == 0) {
                        atomicAdd(demand + IDX(l, x, y), K);
                    }
                    else if(x == X - 1)
                        atomicAdd(demand + IDX(l, x - 1, y), K);
                    else {
                        atomicAdd(demand + IDX(l, x - 1, y), 0.5 * K);
                        atomicAdd(demand + IDX(l, x, y), 0.5 * K);
                    }
                }
                //timestamp[IDX(l, x, y)] = stamp;
            }
        }
    }
    atomicAdd(&total_via_count, via_count * K);
}

__global__ void extract_congestionView() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int x = idx / Y % X, y = idx % Y;
    double resource = capacity[idx] - demand[idx];
    if(resource < 0)
    {
        congestion[x * Y + y] = true;
        congestion_xsum[x * Y + y] = max(congestion_xsum[x * Y + y], -resource);
        congestion_ysum[x * Y + y] = max(congestion_ysum[x * Y + y], -resource);
    }
}

__global__ void extract_congestionView_ysum() {
    extern __shared__ float sum3[];
    if(threadIdx.x == 0) sum3[0] = 0;
    int x = blockIdx.x; // % XY;
    for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum3[y + 1] = congestion_ysum[x*Y+y];
    __syncthreads();
    for(int d = 0; (1 << d) < Y; d++) {
        for(int idx = threadIdx.x; idx < Y; idx += blockDim.x) 
            if(idx >> d & 1) sum3[idx] += sum3[(idx >> d << d) - 1];
        __syncthreads();
    }
    for(int y = threadIdx.x; y < Y; y += blockDim.x) congestion_ysum[x*Y+y] = sum3[y];
}

__global__ void extract_congestionView_xsum() {
    extern __shared__ float sum3[];
    if(threadIdx.x == 0) sum3[0] = 0;
    int y = blockIdx.x;// % XY;
    for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum3[x + 1] = congestion_xsum[x*Y+y];
    __syncthreads();
    for(int d = 0; (1 << d) < X; d++) {
        for(int idx = threadIdx.x; idx < X; idx += blockDim.x) 
            if(idx >> d & 1) sum3[idx] += sum3[(idx >> d << d) - 1];
        __syncthreads();
    }
    for(int x = threadIdx.x; x < X; x += blockDim.x) congestion_xsum[x*Y+y] = sum3[x];
}

vector<bool> is_of_net_cpu;

pair<vector<int>, vector<int>> ripup(int of_threshold) {
    assert(NET_NUM <= db::MAX_NET_NUM);
    static bool *is_of_net_cpu = new bool[NET_NUM]; 
    
    cudaMemset(of_edge_sum, 0, sizeof(int) * L * X * Y);
    mark_overflow_edges<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> (of_threshold);
    compute_presum_general<<<all_track_cnt, THREAD_NUM, sizeof(int) * XY>>> (of_edge_sum);
    mark_overflow_nets<<<BLOCK_NUM(NET_NUM), THREAD_NUM>>> ();
    cudaMemcpy(is_of_net_cpu, is_of_net, sizeof(bool) * NET_NUM, cudaMemcpyDeviceToHost);
    vector<int> of_nets, no_of_nets;
    for(int i = 0; i < NET_NUM; i++) 
        (is_of_net_cpu[i] ? of_nets : no_of_nets).emplace_back(i);
    return make_pair(move(of_nets), move(no_of_nets));
}

}
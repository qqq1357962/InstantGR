#include "graph.hpp"

namespace Lshape_route {

//declaration
void Lshape_route(vector<int> &nets2route);


__global__ void Lshape_route_cuda(int net_cnt, int net_offset, int *node_cnt_sum, int *nodes, int *par_nodes, double *dist, int *from, int *layer_range, int stamp) {
    int net_idx = blockIdx.x * blockDim.x + threadIdx.x;// net_idx-th net in this batch
    if(net_idx >= net_cnt) return;
    net_idx += net_offset;
    int node_cnt = node_cnt_sum[net_idx + 1] - node_cnt_sum[net_idx];// node count of the net
    int *net_routes = routes + pin_acc_num[net_ids[net_idx]] * ROUTE_PER_PIN;
    nodes += node_cnt_sum[net_idx];
    par_nodes += node_cnt_sum[net_idx];
    layer_range += node_cnt_sum[net_idx];
    dist += (node_cnt_sum[net_idx] - node_cnt_sum[net_offset]) * L * L;
    from += (node_cnt_sum[net_idx] - node_cnt_sum[net_offset]) * L * L;
    //compute the via cost for each node and each layer range pair 
    // node_dist[minl * L + maxl] is the total via cost to include layers in [minl, maxl]
    for(int i = 0; i < node_cnt; i++) {
        int l = nodes[i] / X / Y, x = nodes[i] / Y % X, y = nodes[i] % Y;
        double *node_dist = dist + i * L * L;
        for(int minl = 0; minl < L; minl++) {
            node_dist[minl * L + minl] = 0;
            for(int maxl = minl + 1; maxl < L; maxl++)
                node_dist[minl * L + maxl] = node_dist[minl * L + maxl - 1] + vcost[IDX(maxl - 1, x, y)];
            for(int maxl = minl; maxl < L; maxl++)
                if(l < L && (minl > l || maxl < l)) node_dist[minl * L + maxl] = INF;
        }
    }
    for(int i = node_cnt - 1; i >= 1; i--) {
        int x = nodes[i] / Y % X, y = nodes[i] % Y;// current node
        int px = nodes[par_nodes[i]] / Y % X, py = nodes[par_nodes[i]] % Y;// parent node of current node
        int minx = min(x, px), maxx = max(x, px), miny = min(y, py), maxy = max(y, py);
        double *node_dist = dist + i * L * L, *par_dist = dist + par_nodes[i] * L * L;
        assert(par_nodes[i] < i);
        int *prev = from + i * L * L, cur_from[10];
        double min_cost_cur[10], min_cost_par[100];
        for(int l = 0; l < L; l++) {
            min_cost_cur[l] = min_cost_par[l * L + l] = INF;
            for(int minl = 0; minl <= l; minl++)
                for(int maxl = l; maxl < L; maxl++)
                    if(node_dist[minl * L + maxl] < min_cost_cur[l]) {
                        min_cost_cur[l] = node_dist[minl * L + maxl];
                        cur_from[l] = minl * L + maxl;
                    }
        }
        if(x == px || y == py) {
            for(int l = 0; l < L; l++) {
                if((l & 1 ^ DIR) == 0 && y != py) continue;
                if((l & 1 ^ DIR) == 1 && x != px) continue;
                min_cost_par[l * L + l] = min_cost_cur[l] + graph::wire_segment_cost(l, minx, maxx, miny, maxy);//presum[IDX(l, maxx, maxy)] - presum[IDX(l, minx, miny)];
                prev[l * L + l] = cur_from[l] * L * L + l * L + l;
            }
        } else {
            for(int curl = 0; curl < L; curl++) {
                double cost = min_cost_cur[curl];
                if(curl & 1 ^ DIR)
                    cost += graph::wire_segment_cost(curl, x, x, miny, maxy);//presum[IDX(curl, x, maxy)] - presum[IDX(curl, x, miny)];
                else
                    cost += graph::wire_segment_cost(curl, minx, maxx, y, y);//presum[IDX(curl, maxx, y)] - presum[IDX(curl, minx, y)];
                for(int parl = curl & 1 ^ 1; parl < L; parl += 2) {// curl and parl must have different routing directions
                    assert(curl % 2 != parl % 2);
                    double cost2 = 0;
                    for(int l = min(curl, parl) + 1; l < max(curl, parl); l++)
                        cost2 += (curl & 1 ^ DIR) ? vcost[IDX(l, x, py)] : vcost[IDX(l, px, y)];
                    if(parl & 1 ^ DIR)
                        cost2 += graph::wire_segment_cost(parl, px, px, miny, maxy);//presum[IDX(parl, px, maxy)] - presum[IDX(parl, px, miny)];
                    else
                        cost2 += graph::wire_segment_cost(parl, minx, maxx, py, py);//presum[IDX(parl, maxx, py)] - presum[IDX(parl, minx, py)];
                    if(cost + cost2 < min_cost_par[parl * L + parl]) {
                        min_cost_par[parl * L + parl] = cost + cost2;
                        prev[parl * L + parl] = cur_from[curl] * L * L + curl * L + parl;
                    }
                }
            }
        }


        for(int minl = 0; minl < L; minl++) {
            for(int maxl = minl + 1; maxl < L; maxl++) {
                //min_cost_par[minl * L + maxl] = min { min_cost_par[minl * L + maxl - 1], min_cost_par[maxl * L + maxl] }
                if(min_cost_par[minl * L + maxl - 1] <= min_cost_par[maxl * L + maxl]) {
                    min_cost_par[minl * L + maxl] = min_cost_par[minl * L + maxl - 1];
                    prev[minl * L + maxl] = prev[minl * L + maxl - 1];
                } else {
                    min_cost_par[minl * L + maxl] = min_cost_par[maxl * L + maxl];
                    prev[minl * L + maxl] = prev[maxl * L + maxl];
                }
            }
            for(int maxl = minl; maxl < L; maxl++) {
                par_dist[minl * L + maxl] += min_cost_par[minl * L + maxl];
            }
        }
    }

    net_routes[0] = 1;
    layer_range[0] = 0;
    for(int minl = 0; minl < L; minl++)
        for(int maxl = minl; maxl < L; maxl++) 
            if(dist[minl * L + maxl] < dist[layer_range[0]]) {
                layer_range[0] = minl * L + maxl;
            }
    for(int i = 0; i < node_cnt; i++) {
        int *prev = from + i * L * L;
        if(i > 0) layer_range[i] = prev[layer_range[par_nodes[i]]] / L / L;
        assert(dist[i * L * L + layer_range[i]] < INF);
        int minl = layer_range[i] / L, maxl = layer_range[i] % L, x = nodes[i] / Y % X, y = nodes[i] % Y;
        if(minl < maxl) {
            net_routes[net_routes[0]++] = IDX(minl, x, y);
            net_routes[net_routes[0]++] = IDX(maxl, x, y);
        }

        if(i == 0) continue;
        int px = nodes[par_nodes[i]] / Y % X, py = nodes[par_nodes[i]] % Y;
        int minx = min(x, px), maxx = max(x, px), miny = min(y, py), maxy = max(y, py);
        int curl = prev[layer_range[par_nodes[i]]] / L % L, parl = prev[layer_range[par_nodes[i]]] % L;
        if(px == x || py == y) {
            assert(curl == parl);
            net_routes[net_routes[0]++] = IDX(curl, minx, miny);
            net_routes[net_routes[0]++] = IDX(curl, maxx, maxy);
            graph::atomic_add_unit_demand_wire_segment(curl, minx, maxx, miny, maxy, stamp);
        } else {        
            assert(curl % 2 != parl % 2);
            int minl = min(curl, parl), maxl = max(curl, parl);
            if(curl & 1 ^ DIR) {
                net_routes[net_routes[0]++] = IDX(curl, x, y);
                net_routes[net_routes[0]++] = IDX(curl, x, py);
                graph::atomic_add_unit_demand_wire_segment(curl, x, x, miny, maxy, stamp);

                net_routes[net_routes[0]++] = IDX(curl, x, py);
                net_routes[net_routes[0]++] = IDX(parl, x, py);

                net_routes[net_routes[0]++] = IDX(parl, x, py);
                net_routes[net_routes[0]++] = IDX(parl, px, py);
                graph::atomic_add_unit_demand_wire_segment(parl, minx, maxx, py, py, stamp);
            } else {
                net_routes[net_routes[0]++] = IDX(curl, x, y);
                net_routes[net_routes[0]++] = IDX(curl, px, y);
                graph::atomic_add_unit_demand_wire_segment(curl, minx, maxx, y, y, stamp);

                net_routes[net_routes[0]++] = IDX(curl, px, y);
                net_routes[net_routes[0]++] = IDX(parl, px, y);

                net_routes[net_routes[0]++] = IDX(parl, px, y);
                net_routes[net_routes[0]++] = IDX(parl, px, py);
                graph::atomic_add_unit_demand_wire_segment(parl, px, px, miny, maxy, stamp);
            }
        }
    }
}

void Lshape_route(vector<int> &nets2route) {
    double Lshape_start_time = elapsed_time();

    if(LOG) printf("[%5.1f] FLUTE", elapsed_time()); cerr << endl;
    #pragma omp parallel for num_threads(8)
    for(int i = 0; i < nets2route.size(); i++) {
        bool log_flag = false;
        nets[nets2route[i]].construct_rsmt(log_flag);
    }
    if(LOG) printf("[%5.1f] FLUTE END", elapsed_time()); cerr << endl;


    //generate batches
    
    if(LOG) printf("[%5.1f] SORT\n", elapsed_time());
    sort(nets2route.begin(), nets2route.end(), [] (int l, int r) {
        return nets[l].hpwl > nets[r].hpwl;
    });
    if(LOG) printf("[%5.1f] SORT END\n", elapsed_time());

    auto batches = generate_batches_rsmt_gpu(nets2route);
    reverse(batches.begin(), batches.end());

    vector<int> batch_cnt_sum(batches.size() + 1, 0);
    for(int i = 0; i < batches.size(); i++) {
        batch_cnt_sum[i + 1] = batch_cnt_sum[i] + batches[i].size();
        for(int j = 0; j < batches[i].size(); j++)
            nets2route[batch_cnt_sum[i] + j] = batches[i][j];
    }
    int pin_cnt = 0, net_cnt = nets2route.size();
    for(auto net_id : nets2route) pin_cnt += nets[net_id].pins.size();
    int *node_cnt_sum_cpu = new int[net_cnt + 1]();
    int *nodes_cpu = new int[pin_cnt * 2];
    int *par_nodes_cpu = new int[pin_cnt * 2];


    if(LOG) printf("[%5.1f] DFS\n", elapsed_time());
    for(int id = 0; id < net_cnt; id++) {
        auto &graph = nets[nets2route[id]].rsmt;
        node_cnt_sum_cpu[id + 1] = node_cnt_sum_cpu[id] + graph.back().size();
    }
    #pragma omp parallel for num_threads(8)
    for(int id = 0; id < net_cnt; id++) {
        auto &graph = nets[nets2route[id]].rsmt;
        int node_idx_cnt = 0;
        function<void(int, int, int)> dfs = [&] (int x, int par, int par_node_idx) {
            int node_idx = node_idx_cnt++;
            nodes_cpu[node_cnt_sum_cpu[id] + node_idx] = graph.back()[x];
            par_nodes_cpu[node_cnt_sum_cpu[id] + node_idx] = par_node_idx;
            for(auto e : graph[x]) if(e != par) dfs(e, x, node_idx);
        };
        dfs(0, -1, -1);
        assert((node_cnt_sum_cpu[id + 1] - node_cnt_sum_cpu[id]) == graph.back().size());
    }
    if(LOG) printf("[%5.1f] DFS END\n", elapsed_time());
    int max_num_nodes = 0;
    for(int i = 0; i < batches.size(); i++)
        max_num_nodes = max(max_num_nodes, node_cnt_sum_cpu[batch_cnt_sum[i + 1]] - node_cnt_sum_cpu[batch_cnt_sum[i]]);

    double *dist;
    int *from, *layer_range, *node_cnt_sum, *nodes, *par_nodes;

    print_GPU_memory_usage();
    cudaMemcpy(net_ids, nets2route.data(), net_cnt * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&node_cnt_sum, (net_cnt + 1) * sizeof(int));
    cudaMemcpy(node_cnt_sum, node_cnt_sum_cpu, (net_cnt + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&nodes, node_cnt_sum_cpu[net_cnt] * sizeof(int));
    cudaMemcpy(nodes, nodes_cpu, node_cnt_sum_cpu[net_cnt] * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&par_nodes, node_cnt_sum_cpu[net_cnt] * sizeof(int));
    cudaMemcpy(par_nodes, par_nodes_cpu, node_cnt_sum_cpu[net_cnt] * sizeof(int), cudaMemcpyHostToDevice);

    print_GPU_memory_usage();
    cudaMalloc(&dist, max_num_nodes * L * L * sizeof(double));
    cudaMalloc(&from, max_num_nodes * L * L * sizeof(int));
    cudaMalloc(&layer_range, node_cnt_sum_cpu[net_cnt] * sizeof(int));
    
    print_GPU_memory_usage();
    
    if(LOG) printf("[%5.1f] Lshape_cuda\n", elapsed_time());
     for(int i = 0; i < batches.size(); i++) {
        global_timestamp++;
        graph::update_cost();
        graph::compute_presum<<<all_track_cnt, THREAD_NUM, sizeof(double) * XY>>> ();
        Lshape_route_cuda<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), batch_cnt_sum[i], node_cnt_sum, nodes, par_nodes, dist, from, layer_range, global_timestamp);
        graph::batch_wire_update(global_timestamp);
        graph::commit_via_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>> (batches[i].size(), batch_cnt_sum[i], global_timestamp);
    }
    cudaDeviceSynchronize();
    if(LOG) printf("[%5.1f] Lshape_cuda END\n", elapsed_time());

    cudaFree(node_cnt_sum);
    cudaFree(nodes);
    cudaFree(par_nodes);
    cudaFree(dist);
    cudaFree(from);
    cudaFree(layer_range);

    printf("Lshape END.   ");
    print_GPU_memory_usage();

    Lshape_time = elapsed_time() - Lshape_start_time;
}

}
#include "Lshape_route.hpp"
#include "Lshape_route_detour.hpp"
#include "graph.hpp"
#include "database.hpp"
#include "database_cuda.hpp"

void cuda_warmup() {
    int dummy = 0;
    int* d_dummy;
    cudaMalloc(&d_dummy, sizeof(int));
    cudaMemcpy(d_dummy, &dummy, sizeof(int), cudaMemcpyHostToDevice);
    cudaFree(d_dummy);
    cudaDeviceSynchronize();
}
void async_cuda_init() {
    std::thread init_thread([]() {
        cuda_warmup();
    });
    init_thread.detach();
}

void route() {
    build_cuda_database();
    for(int i = 0; i < db::nets.size(); i++)
        if(db::nets[i].pins.size() == 1) nets2output.push(i);
    thread output_thread1(graph::output_nets);
    
    cudaFuncSetAttribute(graph::compute_presum, cudaFuncAttributeMaxDynamicSharedMemorySize, 120 * 1024);
    cudaFuncSetAttribute(graph::compute_presum_general, cudaFuncAttributeMaxDynamicSharedMemorySize, 60 * 1024);
    vector<int> nets2route_all;
    nets2route_all.resize(nets.size());
    for(int i = 0; i < nets.size(); i++) nets2route_all[i] = i;
    Lshape_route::Lshape_route(nets2route_all);
    if(LOG) graph::report_score();
    output_thread1.join();

    int of_threshold = 0;

    if(nets.size() > 5000000) mode = 1;
    if(nets.size() > 20000000) mode = 2;

    graph::extract_congestionView<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>> ();
    graph::extract_congestionView_xsum<<<Y, THREAD_NUM, sizeof(float) * X>>> ();
    graph::extract_congestionView_ysum<<<X, THREAD_NUM, sizeof(float) * Y>>> ();
    auto of_nets = graph::ripup(of_threshold);
    graph::finish_nets(of_nets.second);
    thread output_thread2(graph::output_nets);
    int accu_pin = 0;
    for(int i=0; i < of_nets.first.size(); i++)
    {
        int net_id = of_nets.first[i];
        pin_cnt_sum_phase2_cpu[net_id] = accu_pin;
        accu_pin+=nets[net_id].pins.size();
        ripup_flag_cpu[net_id] = 1;
    }
    PIN_NUM_PHASE2 = accu_pin;
    cudaMallocManaged(&routes_phase2, (ROUTE_PER_PIN_PHASE2 * PIN_NUM_PHASE2 + 1) * sizeof(int));
    cudaMalloc(&ripup_flag, (1 + NET_NUM) * sizeof(int));
    cudaMemcpy(ripup_flag, ripup_flag_cpu.data(), (1 + NET_NUM) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(pin_acc_num_phase2, pin_cnt_sum_phase2_cpu.data(), (1 + NET_NUM) * sizeof(int), cudaMemcpyHostToDevice);
    {
        cudaDeviceSynchronize();
        auto t = cudaGetLastError();
        if (t != cudaSuccess) {
            exit(EXIT_FAILURE);
        }
    }
    Lshape_route_detour::Lshape_route_detour_wrap(of_nets.first);
    if(LOG) graph::report_score();
    output_thread2.join();
    graph::finish_nets(of_nets.first);
    graph::output_nets();
}

void runtime_breakdown() {
    const int width = 20;
    double total_time = elapsed_time();
    cout << fixed << setprecision(2);
    cout << endl << setw(width * 3) << setfill('-') << "-" << setfill(' ') << endl;
    cout << setw(width) << "Procedure" << setw(width) << "time (s)" << setw(width) << "percent (%)" << endl;
    cout << setw(width * 3) << setfill('.') << "." << setfill(' ') << endl;
    cout << setw(width) << "input" << setw(width) << input_time << setw(width) << input_time / total_time * 100 << endl;
    cout << setw(width) << "Lshape Route" << setw(width) << Lshape_time << setw(width) << Lshape_time / total_time * 100 << endl;
    cout << setw(width) << "DAG Route" << setw(width) << DAG_time << setw(width) << DAG_time / total_time * 100 << endl;
    cout << setw(width) << "total" << setw(width) << total_time << setw(width) << total_time / total_time * 100 << endl;
    cout << setw(width * 3) << setfill('-') << "-" << setfill(' ') << endl << endl;
}



int main(int argc, char *argv[]) {
    program_start = std::chrono::high_resolution_clock::now();
    async_cuda_init();
    const int cap_file_idx = 2, net_file_idx = 4, out_file_idx = 6;
    db::read(argv[cap_file_idx], argv[net_file_idx]);
    out_file = fopen(argv[out_file_idx], "w");
    readLUT();
    route();
    fclose(out_file);
    if(LOG) runtime_breakdown();
    quick_exit(0);
}
#pragma once
#include "global.h"
#include "robin_hood.h"
#include "flute.hpp"


namespace db {

struct layer {
    string name;
    int dir;//0: horizontal/X; 1: vertical/Y
    double min_len;
};

struct net {

    void init(int num, vector<vector<int>> &p, int minx, int maxx, int miny, int maxy);
    int minx, maxx, miny, maxy;

    int hpwl, unfinished_subnet_count;
    string name;
    vector<vector<int>> access_points;
    vector<int> pins, extra_routes, subnets;
};

const int MAX_NET_NUM = 90000000;

int L, X, Y;
double unit_length_wire_cost, unit_via_cost;
vector<double> unit_length_short_costs;
vector<int> x_edge_len, y_edge_len, dr_x, dr_y, dr2gr_x, dr2gr_y, real_dr_x, real_dr_y;
vector<layer> layers(10);
vector<vector<vector<double>>> capacity;
vector<net> nets;

tuple<int, int, int> decode_pin(int pin) { return make_tuple(pin / X / Y, pin / Y % X, pin % Y); }

void net::init(int num, vector<vector<int>> &access_points, int xmin, int xmax, int ymin, int ymax) {
    assert(pins.empty());
    hpwl = xmax - xmin + ymax - ymin;
    int center_2x = xmin + xmax, center_2y = ymin + ymax;
    robin_hood::unordered_set<int> p, p2D;

    vector<int> enumerated_idx(num), best_idx(num);
    int min_diff = 1e9, min_metric = 1e9, min_metric2 = 1e9, min_metric0 = 1e9;
    function<void(int)> enumerate = [&] (int cur) {
        if(cur == num) {
            int tmpxmin = 1e9, tmpxmax = 0, tmpymin = 1e9, tmpymax = 0;
            int diff = 0, metric = 0, metric2=0;
            int metric0 = 0;
            for(int i = 0; i < num; i++) {
                int p_i = access_points[i][enumerated_idx[i]];
                tmpxmin = min(tmpxmin, p_i / Y % X);
                tmpxmax = max(tmpxmax, p_i / Y % X);
                tmpymin = min(tmpymin, p_i % Y);
                tmpymax = max(tmpymax, p_i % Y);
                int xi = access_points[i][enumerated_idx[i]] / Y % X, yi = access_points[i][enumerated_idx[i]] % Y;
                int li = access_points[i][enumerated_idx[i]] / X / Y;
                metric += abs(2 * xi - center_2x) + abs(2 * yi - center_2y);
                metric2+=li;
                for(int j = 0; j < i; j++) {
                    int xj = access_points[j][enumerated_idx[j]] / Y % X, yj = access_points[j][enumerated_idx[j]] % Y;
                    diff += (xi != xj) + (yi != yj);
                }
            }
            metric0 = tmpxmax - tmpxmin + tmpymax - tmpymin;
            if(diff < min_diff || (diff == min_diff && metric0 < min_metric0)) {
                best_idx = enumerated_idx;
                min_diff = diff;
                min_metric0 = metric0;
                min_metric = metric;
                min_metric2 = metric2;
            }
            else if(diff == min_diff && metric0 == min_metric0 && metric < min_metric) {
                best_idx = enumerated_idx;
                min_diff = diff;
                min_metric = metric;
                min_metric2 = metric2;
            }
            else if(diff == min_diff && metric0 == min_metric0 && metric == min_metric && metric2 < min_metric2) {
                best_idx = enumerated_idx;
                min_metric2 = metric2;
            }
        } else {
            for(int idx = 1; idx <= access_points[cur][0]; idx++) {
                enumerated_idx[cur] = idx;
                enumerate(cur + 1);
            }
        }
    };

    if(num <= 4) {
        enumerate(0);
        for(int i = 0; i < num; i++) {
            p.insert(access_points[i][best_idx[i]]);
            p2D.insert(access_points[i][best_idx[i]] % (X * Y));
        }
    } else {
        robin_hood::unordered_map<int, std::vector<int>> cnt_pin2d;
        robin_hood::unordered_set<int> might_overlap_pins;
        {
            cnt_pin2d.clear();
            // First pass: Calculate pin overlap counts
            for(int i = 0; i < num; i++) {
                // if(visited_pin[i]) continue;
                for(int j = 1; j <= access_points[i][0]; j++) {
                    int pin = access_points[i][j];
                    int pin_2d = pin%(X*Y);
                    cnt_pin2d[pin_2d].push_back(i);
                }
            }
            for(auto pin_list : cnt_pin2d) {
                for(auto pin_id : pin_list.second) {
                    might_overlap_pins.insert(pin_id);
                }
            }
        }
        
        robin_hood::unordered_map<int, int> x_counts;
        robin_hood::unordered_map<int, int> y_counts;

        std::vector<int> visited_pin(num, 0);
        x_counts.clear();
        y_counts.clear();
        
        for(int round = 0; round < num; round++) {
            int max_pos = -1;
            int cnt_repeat = 0;
            int max_coord_repeat = -1;  // Maximum x/y coordinate repetition with existing pins
            int min_dist = 1e9;
            cnt_pin2d.clear();

            for(auto i: might_overlap_pins) {
                if(visited_pin[i]) continue;
                for(int j = 1; j <= access_points[i][0]; j++) {
                    int pin = access_points[i][j];
                    int pin_2d = pin%(X*Y);
                    cnt_pin2d[pin_2d].push_back(i);
                    // Calculate Manhattan distance to center
                    int dist = abs(pin_2d / Y * 2 - center_2x) + abs(pin_2d % Y * 2 - center_2y);

                    int x_repeat = x_counts[pin_2d/Y], y_repeat = y_counts[pin_2d%Y];
                    int coord_repeat = x_repeat + y_repeat;
                    
                    // Priority 1: overlap count
                    if(cnt_pin2d[pin_2d].size() > cnt_repeat) {
                        max_pos = pin_2d;
                        cnt_repeat = cnt_pin2d[pin_2d].size();
                        max_coord_repeat = coord_repeat;
                        min_dist = dist;
                    }
                    // Priority 2: coordinate repetition (when overlap counts are equal)
                    else if(cnt_pin2d[pin_2d].size() == cnt_repeat && coord_repeat > max_coord_repeat) {
                        max_pos = pin_2d;
                        max_coord_repeat = coord_repeat;
                        min_dist = dist;
                    }
                    // Priority 3: distance to center (when overlap counts and coord repeats are equal)
                    else if(cnt_pin2d[pin_2d].size() == cnt_repeat && 
                            coord_repeat == max_coord_repeat && 
                            dist < min_dist) {
                        max_pos = pin_2d;
                        min_dist = dist;
                    }
                }
            }
            
            if(cnt_repeat <=1) break;
            
            if(!p2D.count(max_pos)) 
            {
                p2D.insert(max_pos);
                int x = max_pos / Y % X;
                int y = max_pos % Y;
                x_counts[x]++;
                y_counts[y]++;
            }
            for(auto pin_id : cnt_pin2d[max_pos]) {
                assert(!visited_pin[pin_id]);
                visited_pin[pin_id] = 1;
                int flag2 = 0;
                for(int j = 1; j <= access_points[pin_id][0]; j++) {
                    int pin = access_points[pin_id][j];
                    if(pin%(X*Y) == max_pos) {
                        p.insert(pin);
                        flag2++;
                        break;
                    }
                }
                assert(flag2 == 1);            
            }
        }

        for(int i = 0; i < num; i++) {
            if(visited_pin[i]) continue;
            int min_dist = 1e9, selected_pin = -1;
            int max_overlap = 0;
            for(int j = 1; j <= access_points[i][0]; j++) {
                int pin = access_points[i][j], dist = abs(pin / Y % X * 2 - center_2x) + abs(pin % Y * 2 - center_2y);
                int pin_2d = pin%(X*Y);
                int x_repeat = x_counts[pin_2d/Y], y_repeat = y_counts[pin_2d%Y];
                int coord_repeat = x_repeat + y_repeat;
                if(coord_repeat > max_overlap) {
                    max_overlap = coord_repeat;
                    min_dist = dist;
                    selected_pin = pin;
                }
                else if(coord_repeat == max_overlap && dist < min_dist) {
                    min_dist = dist;
                    selected_pin = pin;
                }
            }
            assert(selected_pin >= 0);
            p.insert(selected_pin);
            if(!p2D.count(selected_pin % (X * Y))) 
            {
                p2D.insert(selected_pin % (X * Y));
                int x = selected_pin / Y % X;
                int y = selected_pin % Y;
                x_counts[x]++;
                y_counts[y]++;
            }
        }
    }
    if(p2D.size() < p.size()) {
        robin_hood::unordered_map<int, int> pmax, pmin;
        for(auto e : p) {
            int pos_2D = e % (X * Y);
            pmax[pos_2D] = pmax.count(pos_2D) ? max(pmax[pos_2D], e) : e;
            pmin[pos_2D] = pmin.count(pos_2D) ? min(pmin[pos_2D], e) : e;
        }
        for(auto e : p2D) {
            pins.emplace_back(pmax[e]);
            if(pmin[e] != pmax[e]) {
                extra_routes.emplace_back(pmin[e]);
                extra_routes.emplace_back(pmax[e]);
            }
        }
    } else {
        pins.reserve(p.size());
        for(auto e : p) pins.emplace_back(e);
    }
}



void read(char cap_file_name[], char net_file_name[]) {
    double input_start_time = elapsed_time();

    for(int i = 0; i < 10; i++) layers[i].name = "metal" + to_string(i + 1);
    auto read_int = [&] (vector<char> &buffer, size_t &buf_pt) {
        int ans = 0;
        while(!isdigit(buffer[buf_pt])) buf_pt++;
        for(; isdigit(buffer[buf_pt]); buf_pt++) ans = ans * 10 + buffer[buf_pt] - '0';
        return ans;
    };
    auto read_double = [&] (vector<char> &buffer, size_t &buf_pt) {
        double ans = 0, scale = 0;
        while(!isdigit(buffer[buf_pt])) buf_pt++;
        for(; isdigit(buffer[buf_pt]) || buffer[buf_pt] == '.'; buf_pt++)
            if(buffer[buf_pt] == '.') 
                scale = 1;
            else 
                scale *= 10, ans = ans * 10 + buffer[buf_pt] - '0';
        return scale == 0 ? ans : ans / scale;
    };
    auto read_string = [&] (vector<char> &buffer, size_t &buf_pt, string &str) {
        while(buffer[buf_pt] == ' ' || buffer[buf_pt] == '\n') buf_pt++;
        size_t beg = buf_pt;
        while(buffer[buf_pt] != ' ' && buffer[buf_pt] != '\n') buf_pt++;
        str.resize(buf_pt - beg);
        for(size_t i = 0; i < str.size(); i++) str[i] = buffer[beg + i];
    };

    auto read_cap = [&] (char cap_file_name[]) {
        double cap_start_time = elapsed_time();
        std::ifstream cap_file(string(cap_file_name), std::ios::ate);
        if(!cap_file.good()) cout << "failed to open the cap file" << endl;
        size_t fsize = cap_file.tellg();
        cap_file.seekg(0, std::ios::beg);
        std::vector<char> buffer(fsize + 1);
        cap_file.read(buffer.data(), fsize);
        buffer[fsize] = 0;
        size_t buf_pt = 0;

        L = read_int(buffer, buf_pt);
        X = read_int(buffer, buf_pt);
        Y = read_int(buffer, buf_pt);
        unit_length_wire_cost = read_double(buffer, buf_pt);
        unit_via_cost = read_double(buffer, buf_pt);
        unit_length_short_costs.resize(L);
        assert(L == 10);
        capacity = vector<vector<vector<double>>> (L, vector<vector<double>> (X, vector<double> (Y)));
        for(int i = 0; i < L; i++) unit_length_short_costs[i] = read_double(buffer, buf_pt);
        x_edge_len.resize(X - 1);
        y_edge_len.resize(Y - 1);


        for(int i = 0; i < X - 1; i++) x_edge_len[i] = read_int(buffer, buf_pt);
        for(int i = 0; i < Y - 1; i++) y_edge_len[i] = read_int(buffer, buf_pt);
        dr_x = vector<int> (X, 0);
        dr_y = vector<int> (Y, 0);
        for(int i = 0; i < X - 1; i++) dr_x[i + 1] = x_edge_len[i] + dr_x[i];
        for(int i = 0; i < Y - 1; i++) dr_y[i + 1] = y_edge_len[i] + dr_y[i];
        dr2gr_x = vector<int> (dr_x.back() + 1, -1);
        dr2gr_y = vector<int> (dr_y.back() + 1, -1);
        for(int i = 0; i < X; i++) dr2gr_x[dr_x[i]] = i;
        for(int i = 0; i < Y; i++) dr2gr_y[dr_y[i]] = i;

        for(int l = 0; l < L; l++) {
            string name;
            read_string(buffer, buf_pt, name);
            assert(name == layers[l].name);
            layers[l].dir = read_int(buffer, buf_pt);
            layers[l].min_len = read_double(buffer, buf_pt);
            for(int y = 0; y < Y; y++)
                for(int x = 0; x < X; x++)
                    capacity[l][x][y] = read_double(buffer, buf_pt);
            if(l) assert(layers[l].dir != layers[l - 1].dir);
        }

        printf("[%5.1f] read cap file done: duration=%.2fs", elapsed_time(), elapsed_time() - cap_start_time);
        cout << endl;
    };


    auto read_net = [&] (char net_file_name[]) {
        double net_start_time = elapsed_time();
        std::ifstream net_file(string(net_file_name), std::ios::ate);
        if(!net_file.good()) throw std::invalid_argument("failed to open the file '"s + cap_file_name + '\'');    
        size_t fsize = net_file.tellg();
        net_file.seekg(0, std::ios::beg);
        std::vector<char> buffer(fsize + 1);
        net_file.read(buffer.data(), fsize);
        buffer[fsize] = 0;
        size_t buf_pt = 0;

        nets.reserve(60000000);

        vector<vector<int>> access_points(10000, vector<int> (20));
        
        while(1) {//reading a net
            nets.emplace_back(net());
            read_string(buffer, buf_pt, nets.back().name);
            int minx = X, maxx = 0, miny = Y, maxy = 0, pin_id = 0;
            while(1) {//reading a pin (which may have multiple access points)
                if(pin_id >= 10000) access_points.emplace_back(vector<int> (20));
                access_points[pin_id][0] = 0;
                while(1) {//reading an access point
                    int l = read_int(buffer, buf_pt);
                    int x = read_int(buffer, buf_pt);
                    int y = read_int(buffer, buf_pt);
                    minx = min(minx, x);
                    maxx = max(maxx, x);
                    miny = min(miny, y);
                    maxy = max(maxy, y);
                    
                    access_points[pin_id][++access_points[pin_id][0]] = l * X * Y + x * Y + y;
                    assert(access_points[pin_id][0] < 20);
                    bool pin_end = false;
                    while(!isdigit(buffer[buf_pt])) 
                        if(buffer[buf_pt++] == ']') { pin_end = true; break; }
                    if(pin_end) { 
                        pin_id++;
                        break;
                    }
                }
                bool net_end = false;
                while(!isdigit(buffer[buf_pt]))
                    if(buffer[buf_pt++] == ')') { net_end = true; break; }
                if(net_end) {
                    if(nets.back().name=="accu2sc_credit_vld")
                    {
                        for(int i=0; i<pin_id; i++)
                        {
                            printf("pin %d: ", i);
                            for(int j=1; j<=access_points[i][0]; j++)
                            {
                                printf("%d, x: %d, y: %d", access_points[i][j], access_points[i][j] / Y % X, access_points[i][j] % Y);
                            }
                            printf("\n");
                        }
                    }
                    nets.back().minx = minx;
                    nets.back().maxx = maxx;
                    nets.back().miny = miny;
                    nets.back().maxy = maxy;
                    
                    nets.back().access_points.resize(pin_id);
                    for(int i = 0; i < pin_id; i++) {
                        nets.back().access_points[i].reserve(access_points[i][0] + 1);
                        nets.back().access_points[i].resize(access_points[i][0] + 1);
                        memcpy(nets.back().access_points[i].data(), &access_points[i][0],
                               (access_points[i][0] + 1) * sizeof(access_points[i][0]));
                    }
                    break;
                }
            }
            while(buffer[buf_pt] == ' ' || buffer[buf_pt] == '\n') buf_pt++;
            if(buf_pt == fsize) break;
        }
        printf("[%5.1f] read net file done: duration=%.2fs", elapsed_time(), elapsed_time() - net_start_time);
        cout << endl;
        #pragma omp parallel for num_threads(8)
        for(auto &n : nets) {
            n.init(n.access_points.size(), n.access_points, n.minx, n.maxx, n.miny, n.maxy);
        }
        cout << endl;
    };

    read_cap(cap_file_name);
    read_net(net_file_name);

    input_time = elapsed_time() - input_start_time;
}

}
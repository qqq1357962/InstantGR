# InstantGR

InstantGR is a GPU-accelerated global routing tool. 

Check out the following paper for more details.

* Liang Xiao, Shiju Lin, Jinwei Liu, Qinkai Duan, Tsung-Yi Ho, and Evangeline Young, ["InstantGR: Scalable GPU Parallelization for 3-D Global Routing"](https://ieeexplore.ieee.org/document/11015529), IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2025

* Shiju Lin, Liang Xiao, Jinwei Liu and Evangeline Young, ["InstantGR: Scalable GPU Parallelization for Global Routing"](https://shijulin.github.io/files/1239_Final_Manuscript.pdf), ACM/IEEE International Conference on Computer-Aided Design (ICCAD), New Jersey, USA, Oct 27–31, 2024.

## Compile
```bash
cd src
nvcc main.cpp -o ../run/InstantGR -std=c++17 -x cu -O3 -arch=sm_80
```
You may want to change `-arch=sm_80` according to your GPU. For example, if you are running InstantGR on NVIDIA RTX 3090, you need to change it to `-arch=sm_86`.

## Run
```bash
cd run
./InstantGR -cap <cap_file_path> -net <net_file_path> -out <output_path>
```
Example:
```bash
./InstantGR -cap ../benchmarks/mempool_tile_rank.cap -net ../benchmarks/mempool_tile_rank.net -out mempool_tile_rank.out
```

## Evaluate
```bash
cd run
g++ -o evaluator evaluator.cpp -O3 -std=c++17 #compile the evaluator
./evaluator <cap_file_path> <net_file_path> <output_path> # run the evaluator
```
Example:
```bash
./evaluator ../benchmarks/mempool_tile_rank.cap ../benchmarks/mempool_tile_rank.net mempool_tile_rank.out
```

## Benchmarks

The ISPD2024 benchmarks can be downloaded [here](https://drive.google.com/drive/folders/1bon65UEAx8cjSvVhYJ-lgC8QMDX0fvUm).
We provide a small case `mempool_tile_rank` in the folder `benchmarks` for simple testing.

## Contact
[Shiju Lin](https://shijulin.github.io/) (email: sjlin@cse.cuhk.edu.hk)

## License
BSD 3-Clause License

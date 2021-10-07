#include "application.hpp"
#include "instance.hpp"
#include "lower_bounds.hpp"
#include "../algos/algos2D.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;


std::string run_for_instance(const Instance2D & instance, const vector<string> & list_algos)
{
    int LB_cpu = BPP2D_LBcpu(instance);
    int LB_mem = BPP2D_LBmem(instance);
    int LB = std::max(LB_cpu, LB_mem);

    int hint_bin = LB + 500;

    string row(to_string(LB) + "\t");
    string row_time;

    int sol;
    for (const string & algo_name : list_algos)
    {
        AlgoFit2D * algo = createAlgo2D(algo_name, instance);
        if (algo != nullptr)
        {
            auto start = high_resolution_clock::now();
            sol = algo->solveInstance(hint_bin);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - start);

            row.append("\t" + to_string(sol));
            row_time.append("\t" + to_string((float)duration.count() / 1000));

            delete algo;
        }
        else
        {
            cout << "Unknown algo name: " << algo_name << endl;
        }
    }

    row.append(row_time);
    return row;
}


int run_list_algos(string input_path, string& outfile, vector<string>& list_algos, int bin_cpu_capacity, int bin_mem_capacity)
{
    ofstream f(outfile, ios_base::trunc);
    if (!f.is_open())
    {
        cout << "Cannot write file " << outfile << endl;
        return -1;
    }
    cout << "Writing output to file " << outfile << endl;

    // Header line
    string header("instance_name\tLB");
    string time_header;

    for (std::string algo_name : list_algos)
    {
        header.append("\t" + algo_name);
        time_header.append("\t" + algo_name + "_time");
    }
    f << header << time_header << "\n";

    vector<string> densities = { "005"};
    vector<int> sizes = { 10000, 50000, 100000 };
    vector<string> graph_classes = { "arbitrary", "normal", "threshold" };

    for (string& d : densities)
    {
        for (int s : sizes)
        {
            cout << "Starting density " << d << " size: " << to_string(s) << endl;
            for (string& graph_class : graph_classes)
            {
                cout << "Graph class: " << graph_class << endl;
                for (int n = 0; n < 10; ++n)
                {
                    string instance_name("large_scale_" + to_string(s) + "_" + graph_class + "_d" + d + "_" + to_string(n));
                    string infile(input_path + instance_name + ".csv");
                    const Instance2D instance(instance_name, bin_cpu_capacity, bin_mem_capacity, infile);

                    string row_str = run_for_instance(instance, list_algos);
                    f << instance_name << "\t" << row_str << "\n";
                    f.flush();
                }
            }
        }
    }

    f.close();
    return 0;
}


int main(int argc, char** argv)
{
    string data_path;

    int bin_cpu_capacity;
    int bin_mem_capacity;
    if (argc > 3)
    {
        bin_cpu_capacity = stoi(argv[1]);
        bin_mem_capacity = stoi(argv[2]);
        data_path = argv[3];
    }
    else
    {
        cout << "Usage: " << argv[0] << " <bin_cpu_capacity> <bin_mem_capacity> <data_path>" << endl;
        return -1;
    }

    string input_path(data_path+"/input/");
    string output_path(data_path+"/results/");

    string outfile(output_path + "large2D_" + to_string(bin_cpu_capacity) + "_" + to_string(bin_mem_capacity) + ".csv");

    vector<string> list_algos = {
        "FF", "FFD-Degree",

        "FFD-Avg", "FFD-Max",
        "FFD-AvgExpo", "FFD-Surrogate",
        "FFD-ExtendedSum",

        "BFD-Avg", "BFD-Max",
        "BFD-AvgExpo", "BFD-Surrogate",
        "BFD-ExtendedSum",

        "FFD-DotProduct", "FFD-Fitness",
    };

    run_list_algos(input_path, outfile, list_algos, bin_cpu_capacity, bin_mem_capacity);

    std::cout << "Run successful!" << std::endl;
    return 0;
}



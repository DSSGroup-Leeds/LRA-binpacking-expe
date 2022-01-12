#include "application.hpp"
#include "instance.hpp"
#include "lower_bounds.hpp"
#include "../algos/algosTS.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;


std::string run_for_instance(const InstanceTS & instance,
                             const vector<string> & list_algos)
{
    int LB_cpu, LB_mem;
    TS_LB(instance, LB_cpu, LB_mem);
    int LB = std::max(LB_cpu, LB_mem);
    int hint_bin = LB + 500;

    int best_sol = instance.getTotalReplicas();
    string best_algo;

    string row(to_string(LB));
    string row_res;
    string row_time;

    int sol;
    for (const string & algo_name : list_algos)
    {
        AlgoFitTS * algo = createAlgoTS(algo_name, instance);
        if (algo != nullptr)
        {
            auto start = high_resolution_clock::now();
            sol = algo->solveInstance(hint_bin);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            if (sol < best_sol)
            {
                best_sol = sol;
                best_algo = algo_name;
            }

            row_res.append("\t" + to_string(sol));
            row_time.append("\t" + to_string((float)duration.count()));

            cout << algo_name << " " << to_string(sol) << " " << to_string((float)duration.count()) << endl;

            delete algo;
        }
        else
        {
            cout << "Unknown algo name: " << algo_name << endl;
        }
    }

    row.append("\t"+to_string(best_sol) + "\t" + best_algo);
    //row.append(row_time);
    return row + row_res + row_time;
}


int run_list_algos(string input_path, string& outfile,
                   vector<string>& list_algos,
                   int bin_cpu_capacity, int bin_mem_capacity,
                   int ssize, string graph)
{
    ofstream f(outfile, ios_base::trunc);
    if (!f.is_open())
    {
        cout << "Cannot write file " << outfile << endl;
        return -1;
    }
    cout << "Writing output to file " << outfile << endl;

    // Header line
    string header("instance_name\tLB\tbest_sol\tbest_algo");
    string time_header;

    for (std::string algo_name : list_algos)
    {
        header.append("\t" + algo_name);
        time_header.append("\t" + algo_name + "_time");
    }
    f << header << time_header << "\n";

    vector<string> densities = { "005"};
    vector<int> sizes;// = { 10000, 50000, 100000 };
    sizes.push_back(ssize);
    vector<string> graph_classes;// = { "arbitrary", "normal", "threshold" };
    graph_classes.push_back(graph);

    size_t size_series = 98;

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
                    const InstanceTS instance(instance_name, bin_cpu_capacity, bin_mem_capacity, infile, size_series);

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
    string input_path = "/nobackup/scscm/TClab_data/largeTS/";
    string output_path = "/nobackup/scscm/output/";

    int bin_cpu_capacity;
    int bin_mem_capacity;
    int size;
    string graph;
    if (argc > 4)
    {
        bin_cpu_capacity = stoi(argv[1]);
        bin_mem_capacity = stoi(argv[2]);
        size = stoi(argv[3]);
        graph = argv[4];
    }
    else
    {
        cout << "Usage: " << argv[0] << " <bin_cpu_capacity> <bin_mem_capacity> <size> <graph_class>" << endl;
        return -1;
    }

    string outfile(output_path + "largeTS_" + graph + "_" + to_string(bin_cpu_capacity) + "_" + to_string(bin_mem_capacity) + "_" + to_string(size) + ".csv");
    //string outfile(output_path + "largeTS_" + to_string(bin_cpu_capacity) + "_" + to_string(bin_mem_capacity) + ".csv");

    vector<string> list_algos = {
        "FF", "FFD-Degree",
        "BFD-Avg",
        "WFD-AvgExpo",
    };

    run_list_algos(input_path, outfile, list_algos, bin_cpu_capacity, bin_mem_capacity, size, graph);

    return 0;
}



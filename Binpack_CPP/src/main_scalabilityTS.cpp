#include "application.hpp"
#include "instance.hpp"
#include "lower_bounds.hpp"
#include "../algos/algosTS.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

std::string run_for_instance(const InstanceTS & instance, const vector<string> & list_algos)
{
    int LB_cpu, LB_mem;
    TS_LB(instance, LB_cpu, LB_mem);
    
    int LB = std::max(LB_cpu, LB_mem);
    int hint_bin = LB + 500;

    string row(to_string(LB));
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

    vector<int> nb_apps = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    vector<string> graph_classes = { "arbitrary", "normal", "threshold" };
    std::string d = "005";

    size_t size_series = 98;

    for (int s : nb_apps)
    {
        int nb = s * 10000;
        for (string& graph_class : graph_classes)
        {
            cout << "Starting n " << to_string(nb) << " graph class: " << graph_class << endl;
            
            string instance_name("scalability_" + to_string(nb) + "_" + graph_class + "_d" + d);
            string infile(input_path + instance_name + ".csv");
            const InstanceTS instance(instance_name, bin_cpu_capacity, bin_mem_capacity, infile, size_series);

            string row_str = run_for_instance(instance, list_algos);
            f << instance_name << "\t" << row_str << "\n";
            
            f.flush();
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

    string outfile(output_path + "scalabilityTS_" + to_string(bin_cpu_capacity) + "_" + to_string(bin_mem_capacity) + ".csv");

    vector<string> list_algos = {
        "FF", "FFD-Degree",
        "FFD-Avg", "BFD-Avg",
    };

    run_list_algos(input_path, outfile, list_algos, bin_cpu_capacity, bin_mem_capacity);

    std::cout << "Run successful!" << std::endl;

    return 0;
}


#include "application.hpp"
#include "instance.hpp"
#include "lower_bounds.hpp"
#include "../algos/algos2D.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;


std::string run_for_instance(const Instance2D & instance,
                             const vector<string> & list_algos,
                             const vector<string> & list_spread)
{
    int LB_cpu = BPP2D_LBcpu(instance);
    int LB_mem = BPP2D_LBmem(instance);
    int LB = std::max(LB_cpu, LB_mem);

    int hint_bin = LB + 500;
    int best_sol = instance.getTotalReplicas();


    string row(to_string(LB));
    string row_res;
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

            if (sol < best_sol)
            {
                best_sol = sol;
            }

            row_res.append("\t" + to_string(sol));
            row_time.append("\t" + to_string((float)duration.count() / 1000));

            delete algo;
        }
        else
        {
            cout << "Unknown algo name: " << algo_name << endl;
        }
    }

    row.append("\t"+to_string(best_sol));

    for (const string & algo_name : list_spread)
    {
        Algo2DSpreadWFAvg * algo = createSpreadAlgo(algo_name, instance);
        if (algo != nullptr)
        {
            auto start = high_resolution_clock::now();
            sol = algo->solveInstanceSpread(LB, best_sol);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(stop - start);

            if ((sol < best_sol) and (sol != -1))
            {
                best_sol = sol;
            }

            row_res.append("\t" + to_string(sol));
            row_time.append("\t" + to_string((float)duration.count() / 1000));

            delete algo;
        }
        else
        {
            cout << "Unknown algo name: " << algo_name << endl;
        }
    }

    row.append("\t"+to_string(best_sol));
    return row + row_res + row_time;
}


int run_list_algos(string input_path, string& outfile,
                   vector<string>& list_algos, vector<string>& list_spread,
                   int bin_cpu_capacity, int bin_mem_capacity,
                   int density)
{
    ofstream f(outfile, ios_base::trunc);
    if (!f.is_open())
    {
        cout << "Cannot write file " << outfile << endl;
        return -1;
    }
    cout << "Writing output to file " << outfile << endl;

    // Header line
    string header("instance_name\tLB\tbest_sol\tbest_spread");
    string time_header;

    for (std::string algo_name : list_algos)
    {
        header.append("\t" + algo_name);
        time_header.append("\t" + algo_name + "_time");
    }
    for (std::string algo_name : list_spread)
    {
        header.append("\t" + algo_name);
        time_header.append("\t" + algo_name + "_time");
    }
    f << header << time_header << "\n";

    vector<int> densities;// = { 1, 5, 10 };
    densities.push_back(density);
    vector<string> graph_classes = { "arbitrary", "normal", "threshold" };

    for (int d : densities)
    {
        for (string& graph_class : graph_classes)
        {
            cout << "Starting density " << to_string(d) << " graph class: " << graph_class << endl;
            for (int n = 0; n < 10; ++n)
            {
                cout << to_string(n) << " ";
                string instance_name(graph_class + "_d" + to_string(d) + "_" + to_string(n));
                string infile(input_path + instance_name + ".csv");
                const Instance2D instance(instance_name, bin_cpu_capacity, bin_mem_capacity, infile);

                string row_str = run_for_instance(instance, list_algos, list_spread);
                f << instance_name << "\t" << row_str << "\n";
                f.flush();
            }
            cout << endl;
        }
    }

    f.close();
    return 0;
}


int main(int argc, char** argv)
{
    string input_path = "/nobackup/scscm/TClab_data/density_2D/";
    string output_path = "/nobackup/scscm/output/";
    //string data_path = "/home/mommess/Documents/Leeds_research/datasets/scheduler_trace_datasets/datasets/TClab_data/";

    int bin_cpu_capacity;
    int bin_mem_capacity;
    int density;
    if (argc > 3)
    {
        bin_cpu_capacity = stoi(argv[1]);
        bin_mem_capacity = stoi(argv[2]);
        density = stoi(argv[3]);
    }
    else
    {
        cout << "Usage: " << argv[0] << " <bin_cpu_capacity> <bin_mem_capacity> <density>" << endl;
        return -1;
    }

    //string input_path(data_path+"/input/");
    //string output_path(data_path+"/results/");
    //string input_path(data_path+"large_scale/large_2D/");
    //string outfile("test.csv");

    string outfile(output_path + "density2D_" + to_string(bin_cpu_capacity) + "_" + to_string(bin_mem_capacity) + "_" + to_string(density) + ".csv");

    vector<string> list_algos = {
        "FF",
        //"FFD-Degree",

        //"FFD-Avg", "FFD-Max",
        "FFD-CPU",
        //"FFD-AvgExpo", "FFD-Surrogate",
        //"FFD-ExtendedSum",

        //"BFD-Avg", "BFD-Max",
        "BFD-CPU",
        "BFD-AvgExpo", "BFD-Surrogate",
        "BFD-ExtendedSum",

        "WFD-Avg", "WFD-Max",
        "WFD-CPU",
        "WFD-AvgExpo", "WFD-Surrogate",
        "WFD-ExtendedSum",

        "NCD-L2Norm",
        "NCD-DotProduct", "NCD-Fitness",
        "NCD-DotDivision", "NCD-FitnessDiv"
        //"NodeCount",
    };

    vector<string> list_spread = {
        "SpreadWF-Avg",
        "SpreadWF-Max",
        "SpreadWF-AvgExpo",
        "SpreadWF-Surrogate",
        "SpreadWF-ExtendedSum",
    };

    run_list_algos(input_path, outfile, list_algos, list_spread, bin_cpu_capacity, bin_mem_capacity, density);
    cout << "Run successful" << endl;
    return 0;
}



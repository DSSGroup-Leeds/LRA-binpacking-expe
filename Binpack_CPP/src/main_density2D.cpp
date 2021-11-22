#include "application.hpp"
#include "instance.hpp"
#include "lower_bounds.hpp"
#include "../algos/algos2D.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;


/*std::string run_for_instance(const Instance2D & instance, const vector<string> & list_algos)
{
    int LB_cpu = BPP2D_LBcpu(instance);
    int LB_mem = BPP2D_LBmem(instance);
    int LB = std::max(LB_cpu, LB_mem);

    int hint_bin = LB + 500;

    string row(to_string(LB));
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

    vector<int> densities = { 1, 5, 10 };
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

                string row_str = run_for_instance(instance, list_algos);
                f << instance_name << "\t" << row_str << "\n";
                f.flush();
            }
            cout << endl;
        }
    }

    f.close();
    return 0;
}*/


int main(int argc, char** argv)
{
    //string data_path;
    string data_path = "/home/mommess/Documents/Leeds_research/datasets/scheduler_trace_datasets/datasets/TClab_data/";

    int bin_cpu_capacity;
    int bin_mem_capacity;
    if (argc > 2)
    {
        bin_cpu_capacity = stoi(argv[1]);
        bin_mem_capacity = stoi(argv[2]);
    }


    string instance_name("application_dataset_full");
    string filename = data_path + instance_name + ".csv";
    //string instance_name("arbitrary_d1_0");
    //string filename = data_path + "high_density/2D/" + instance_name + ".csv";
    Instance2D instance(instance_name, bin_cpu_capacity, bin_mem_capacity, filename);

    std::cout << instance_name << std::endl;

    int LB_cpu = BPP2D_LBcpu(instance);
    int LB_mem = BPP2D_LBmem(instance);
    int LB = std::max(LB_cpu, LB_mem);

    Algo2DFF * algoFF = new Algo2DFF(instance);
    int FF_bins = algoFF->solveInstance(LB);

    std::cout << "LB: " << LB << "\nFF: " << FF_bins << std::endl;

    vector<string> list_algos = {
        //"FF",
        "FFD-Degree",

        //"FFD-Avg", "FFD-Max",
        "FFD-CPU",
        //"FFD-AvgExpo", "FFD-Surrogate",
        //"FFD-ExtendedSum",

        "BFD-Avg", "BFD-Max",
        "BFD-CPU",
        "BFD-AvgExpo", "BFD-Surrogate",
        "BFD-ExtendedSum",

        //"FFD-L2Norm", "FFD-DotProduct", "FFD-Fitness",
        "WFD-Avg",
        "WFD-Max",
        "WFD-CPU",
        "WFD-AvgExpo", "WFD-Surrogate",
        "WFD-ExtendedSum",
        //"NodeCount",
    };

    vector<string> list_spread = {
        "SpreadWF-Avg",
        "SpreadWF-Max",
        "SpreadWF-AvgExpo",
        "SpreadWF-Surrogate",
        "SpreadWF-ExtendedSum",
    };

    int best = FF_bins;
    for (string & algo_name : list_algos)
    {
        AlgoFit2D * algo = createAlgo2D(algo_name, instance);
        auto start = high_resolution_clock::now();
        int sol = algo->solveInstance(LB+1000);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << algo_name << ": " << sol << " in " << to_string((float)duration.count() / 1000) << endl;

        if (sol < best)
        {
            best = sol;
        }
    }

    for (string & algo_name : list_spread)
    {
        Algo2DSpreadWFAvg * algo = createSpreadAlgo(algo_name, instance);
        auto start = high_resolution_clock::now();
        int sol = algo->solveInstanceSpread(LB, best);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << algo_name << ": " << sol << " in " << to_string((float)duration.count() / 1000) << endl;
        if ((sol < best) and (sol != -1))
        {
            best = sol;
        }
    }


    /*if (argc > 3)
    {
        bin_cpu_capacity = stoi(argv[1]);
        bin_mem_capacity = stoi(argv[2]);
        data_path = argv[3];
    }
    else
    {
        cout << "Usage: " << argv[0] << " <bin_cpu_capacity> <bin_mem_capacity> <data_path>" << endl;
        return -1;
    }*/

    /*

    TODO

    - Fix measure computation for L2Norm: there is a minus sign
      because we want the app with the smallest value of the measure.

    - Try to change the main Fit algorithm to put as much replicas as possible
      when an app and a node is selected.
      (for the moment we put a single replica and then recompute the measures
       and sort the bins)

    - Try to optimise node-centric approach:
      only compute the measure of apps if they can be packed into the current bin.
      (cost of feasibility check is smaller than cost of computing the measure)

    */

    /*string input_path(data_path+"/input/");
    string output_path(data_path+"/results/");

    string outfile(output_path + "density2D_" + to_string(bin_cpu_capacity) + "_" + to_string(bin_mem_capacity) + ".csv");

    vector<string> list_algos = {
        "FF", "FFD-Degree",
        
        "FFD-Avg", "FFD-Max",
        "FFD-AvgExpo", "FFD-Surrogate",
        "FFD-ExtendedSum",
        
        "BFD-Avg", "BFD-Max",
        "BFD-AvgExpo", "BFD-Surrogate",
        "BFD-ExtendedSum",

        "FFD-L2Norm", "FFD-DotProduct", "FFD-Fitness",
        "NodeCount",
    };

    run_list_algos(input_path, outfile, list_algos, bin_cpu_capacity, bin_mem_capacity);
    cout << "Run successful" << endl;*/
    return 0;
}



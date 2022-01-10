#include "instance.hpp"
#include "lower_bounds.hpp"
#include "algos/algos2D.hpp"

#include "application.hpp"
#include <gurobi_c++.h>
#include <cmath>

#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

int solve_model(GRBModel &model, float &time, int &obj_val, int &best_bound)
{
    // Optimize model
    auto start = high_resolution_clock::now();
    model.optimize();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    time = ((float)duration.count()) / 1000.0;

    obj_val = -1;
    best_bound = -1;
    int optim_status = model.get(GRB_IntAttr_Status);
    if (optim_status == GRB_OPTIMAL)
    {
        obj_val = model.get(GRB_DoubleAttr_ObjVal);
        best_bound = model.get(GRB_DoubleAttr_ObjBound);
    } else if (optim_status == GRB_TIME_LIMIT)
    {
        obj_val = model.get(GRB_DoubleAttr_ObjVal);
        best_bound = model.get(GRB_DoubleAttr_ObjBound);

    } else if ((optim_status == GRB_INF_OR_UNBD) or
                (optim_status == GRB_INFEASIBLE) or
                (optim_status == GRB_UNBOUNDED) )
    {
        std::cout << "Model is infeasible or unbounded" << std::endl;
    } else {
        std::cout << "Optimization was stopped with status = " << optim_status << std::endl;
    }
    return optim_status;
}


int get_tab_index(int j, int i, int nb_bins)
{
    return j * nb_bins + i;
}


GRBModel generate_ILP_2D(GRBEnv env, const Instance2D &instance,
                         int bin_cpu_capacity, int bin_mem_capacity,
                         int nb_bins)
{
    const AppList2D & item_list = instance.getApps();
    int nb_items = item_list.size();

    // Create an empty model
    GRBModel model(env);

    // Create variables
    // y_i: whether bin i is used
    GRBVar* yvars = model.addVars(nb_bins, GRB_BINARY);
    // r_j_i: how many replicas of item j in bin i
    GRBVar* rvars = model.addVars(nb_bins*nb_items, GRB_INTEGER);
    // z_j_i: whether there are replicas of item j in bin i
    GRBVar* zvars = model.addVars(nb_bins*nb_items, GRB_BINARY);
    // Index of var[j,i] is j * nb_bins + i (computed via function get_tab_index)

    int n = 0;

    GRBLinExpr objective = 0;
    for (int i = 0; i < nb_bins; ++i)
    {
        if ( (i%100) == 0)
        {
            std::cout << "Bin: " << i << " constraints: " << n << std::endl;
        }

        objective += yvars[i];

        // Constraint bin capacity
        GRBLinExpr bin_cpu_cap_i = 0;
        GRBLinExpr bin_mem_cap_i = 0;

        for (int j = 0; j < nb_items; ++j)
        {
            Application2D* item = item_list[j];
            bin_cpu_cap_i += (item->getCPUSize()) * rvars[get_tab_index(j,i,nb_bins)];
            bin_mem_cap_i += (item->getMemorySize()) * rvars[get_tab_index(j,i,nb_bins)];

            // Constraint of replicas in a bin and z_ji
            int ceil_cpu_value = ceil(bin_cpu_capacity / (item->getCPUSize()));
            int ceil_mem_value = ceil(bin_mem_capacity / (item->getMemorySize()));
            int max_rep_j = std::min(item->getNbReplicas(), std::min(ceil_cpu_value, ceil_mem_value));
            model.addConstr(rvars[get_tab_index(j,i,nb_bins)] <= max_rep_j * zvars[get_tab_index(j,i,nb_bins)], "(4)");
            n++;

            // Constraints on affinities of other items
            for (auto it_k : item->getAffinityOutMap())
            {
                // For each item in the affinity out map add a constraint
                // rvars_k,i <= aff_value_j_k * z_j_i
                Application2D* item_k = getApp2D(item_list, it_k.first);

                int ceil_cpu_value_k = ceil(bin_cpu_capacity / (item_k->getCPUSize()));
                int ceil_mem_value_k = ceil(bin_mem_capacity / (item_k->getMemorySize()));
                int max_rep_k = std::min(item_k->getNbReplicas(), std::min(ceil_cpu_value_k, ceil_mem_value_k));

                model.addConstr(rvars[get_tab_index(item_k->getInternalId(), i, nb_bins)] <= max_rep_k*(1-zvars[get_tab_index(j, i, nb_bins)]) + ((it_k.second) * zvars[get_tab_index(j,i,nb_bins)]) , "(5)");
                n++;
            }
        }
        model.addConstr(bin_cpu_cap_i <= bin_cpu_capacity * yvars[i], "(3)");
        model.addConstr(bin_mem_cap_i <= bin_mem_capacity * yvars[i], "(3')");
        n+=2;
    }

    // Constraints for all replicas allocated
    for (Application2D* item : item_list)
    {
        GRBLinExpr total_rep_j = 0;
        for (int i = 0; i < nb_bins; ++i)
        {
            total_rep_j += rvars[get_tab_index(item->getInternalId(), i, nb_bins)];

            // r_ji >= z_ji
            model.addConstr(rvars[get_tab_index(item->getInternalId(), i, nb_bins)] >= zvars[get_tab_index(item->getInternalId(), i, nb_bins)], "(6)");
        }

        model.addConstr(total_rep_j >= (item->getNbReplicas()), "(1)");
    }

    model.setObjective(objective, GRB_MINIMIZE);
    model.update();

    delete yvars;
    delete rvars;
    delete zvars;

    return model;
}

GRBModel generate_ILP_2D_noaff(GRBEnv env, const Instance2D &instance,
                         int bin_cpu_capacity, int bin_mem_capacity,
                         int nb_bins)
{
    const AppList2D & item_list = instance.getApps();
    int nb_items = item_list.size();

    // Create an empty model
    GRBModel model(env);

    // Create variables
    // y_i: whether bin i is used
    GRBVar* yvars = model.addVars(nb_bins, GRB_BINARY);
    // r_j_i: how many replicas of item j in bin i
    GRBVar* rvars = model.addVars(nb_bins*nb_items, GRB_INTEGER);
    // z_j_i: whether there are replicas of item j in bin i
    GRBVar* zvars = model.addVars(nb_bins*nb_items, GRB_BINARY);
    // Index of var[j,i] is j * nb_bins + i (computed via function get_tab_index)

    int n = 0;

    GRBLinExpr objective = 0;
    for (int i = 0; i < nb_bins; ++i)
    {
        if ( (i%100) == 0)
        {
            std::cout << "Bin: " << i << " constraints: " << n << std::endl;
        }
        objective += yvars[i];

        // Constraint bin capacity
        GRBLinExpr bin_cpu_cap_i = 0;
        GRBLinExpr bin_mem_cap_i = 0;

        for (int j = 0; j < nb_items; ++j)
        {
            Application2D* item = item_list[j];
            bin_cpu_cap_i += (item->getCPUSize()) * rvars[get_tab_index(j,i,nb_bins)];
            bin_mem_cap_i += (item->getMemorySize()) * rvars[get_tab_index(j,i,nb_bins)];

            // Constraint of replicas in a bin and z_ji
            int ceil_cpu_value = ceil(bin_cpu_capacity / (item->getCPUSize()));
            int ceil_mem_value = ceil(bin_mem_capacity / (item->getMemorySize()));
            int max_rep_j = std::min(item->getNbReplicas(), std::min(ceil_cpu_value, ceil_mem_value));
            model.addConstr(rvars[get_tab_index(j,i,nb_bins)] <= max_rep_j * zvars[get_tab_index(j,i,nb_bins)], "(4)");
            n++;

            // Constraints on affinities of other items
            /*for (auto it_k : item->getAffinityOutMap())
            {
                // For each item in the affinity out map add a constraint
                // rvars_k,i <= aff_value_j_k * z_j_i
                Application2D* item_k = getApp2D(item_list, it_k.first);

                int ceil_cpu_value_k = ceil(bin_cpu_capacity / (item_k->getCPUSize()));
                int ceil_mem_value_k = ceil(bin_mem_capacity / (item_k->getMemorySize()));
                int max_rep_k = std::min(item_k->getNbReplicas(), std::min(ceil_cpu_value_k, ceil_mem_value_k));

                model.addConstr(rvars[get_tab_index(item_k->getInternalId(), i, nb_bins)] <= max_rep_k*(1-zvars[get_tab_index(j, i, nb_bins)]) + ((it_k.second) * zvars[get_tab_index(j,i,nb_bins)]) , "(5)");
            }*/
        }
        model.addConstr(bin_cpu_cap_i <= bin_cpu_capacity * yvars[i], "(3)");
        model.addConstr(bin_mem_cap_i <= bin_mem_capacity * yvars[i], "(3')");
        n+=2;
    }

    // Constraints for all replicas allocated
    for (Application2D* item : item_list)
    {
        GRBLinExpr total_rep_j = 0;
        for (int i = 0; i < nb_bins; ++i)
        {
            total_rep_j += rvars[get_tab_index(item->getInternalId(), i, nb_bins)];

            // r_ji >= z_ji
            model.addConstr(rvars[get_tab_index(item->getInternalId(), i, nb_bins)] >= zvars[get_tab_index(item->getInternalId(), i, nb_bins)], "(6)");
            n++;
        }

        model.addConstr(total_rep_j >= (item->getNbReplicas()), "(1)");
        n++;
    }

    model.setObjective(objective, GRB_MINIMIZE);
    model.update();

    std::cout << "Total constraints: " << n << std::endl;

    delete yvars;
    delete rvars;
    delete zvars;

    return model;
}



int main(int argc, char** argv)
{
    string input_path = "/nobackup/scscm/TClab_data/density2D/";
    string output_path = "/nobackup/scscm/new_outputs/";

    int bin_cpu_capacity = 64;
    int bin_mem_capacity = 128;
    int max_size = 100000;
    if (argc > 1)
    {
        max_size = atoi(argv[1]);
    }
    /*int density;
    if (argc > 3)
    {
        bin_cpu_capacity = stoi(argv[1]);
        bin_mem_capacity = stoi(argv[2]);
        density = stoi(argv[3]);
    }*/
    /*if (argc > 1)
    {
        instance_name = argv[1];
    }
    else
    {
        cout << "Usage: " << argv[0] << " <instance_name>" << endl;
        //cout << "Usage: " << argv[0] << " <bin_cpu_capacity> <bin_mem_capacity> <instance_name>" << endl;
        return -1;
    }*/

    string time_limit("14400");
    cout << "Gurobi time limit is " << time_limit << " seconds" << endl;

    try {
        GRBEnv env(true);
        //env.set("LogFile", 0);
        //env.set("OutputFlag", "0"); // To shut down all logs (LogFile + LogToConsole)
        env.set("DisplayInterval", "600");
        //env.set("Threads", "4");
        //env.set("Method", "3");
        env.set("TimeLimit", time_limit);
        env.start();

        //string instance_name = "arbitrary_d1_0";
        //string infile(input_path + instance_name + ".csv");

        string instance_name = "full_dataset2D";
        //string instance_name = "dataset_2500_1";
        string infile(input_path + "../" + instance_name + ".csv");

        const Instance2D instance(instance_name, bin_cpu_capacity, bin_mem_capacity, infile, max_size);

        cout << "Total apps: " << instance.getApps().size() << endl;
        cout << "Total replicas: " << instance.getTotalReplicas() << endl;

        int LB = BPP2D_LB(instance);

        // Need an upper bound for the ILP
        Algo2DFF * algoFF = new Algo2DFF(instance);
        int ILP_nb_bins = algoFF->solveInstance(LB);

        cout << "LB: " << LB << " FF: " << ILP_nb_bins << endl;

        // Solving ILP with Gurobi
        auto start = high_resolution_clock::now();
        //GRBModel model = generate_ILP_2D_noaff(env, instance, bin_cpu_capacity, bin_mem_capacity, ILP_nb_bins);
        GRBModel model = generate_ILP_2D(env, instance, bin_cpu_capacity, bin_mem_capacity, ILP_nb_bins);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(stop - start);

        cout << "Model generated in " << duration.count() << " seconds" << endl;

        float time;
        int best_bound, obj_value;
        solve_model(model, time, obj_value, best_bound);
        cout << "Obj: " << obj_value << " best bound: " << best_bound << " time: " << time << endl;
    } catch(GRBException e) {
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
    } catch(std::exception e) {
        cout << "Exception during optimization: "<< e.what() << endl;
    }
    
    //string outfile(output_path + "gurobi2D_" + to_string(bin_cpu_capacity) + "_" + to_string(bin_mem_capacity) + ".csv");

    //run_gurobi(input_path, outfile, bin_cpu_capacity, bin_mem_capacity, time_limit, density);
    return 0;
}

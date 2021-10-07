#include "algosTS.hpp"

#include <algorithm> // For stable_sort
#include <iostream>
#include <unordered_map>
#include <cmath> // For exp

AlgoFitTS* createAlgoTS(const std::string& algo_name, const InstanceTS &instance)
{
    if (algo_name == "FF")
    {
        return new AlgoTSFF(instance);
    }
    else if (algo_name == "FFD-Degree")
    {
        return new AlgoTSFFDDegree(instance);
    }
    else if (algo_name == "FFD-Avg")
    {
        return new AlgoTSFFDAvg(instance);
    }
    else if (algo_name == "FFD-Max")
    {
        return new AlgoTSFFDMax(instance);
    }
    else if (algo_name == "FFD-AvgExpo")
    {
        return new AlgoTSFFDAvgExpo(instance);
    }
    else if (algo_name == "FFD-Surrogate")
    {
        return new AlgoTSFFDSurrogate(instance);
    }
    else if (algo_name == "FFD-ExtendedSum")
    {
        return new AlgoTSFFDExtendedSum(instance);
    }

    else if(algo_name == "BFD-Avg")
    {
        return new AlgoTSBFDAvg(instance);
    }
    else if(algo_name == "BFD-Max")
    {
        return new AlgoTSBFDMax(instance);
    }
    else if(algo_name == "BFD-AvgExpo")
    {
        return new AlgoTSBFDAvgExpo(instance);
    }
    else if(algo_name == "BFD-Surrogate")
    {
        return new AlgoTSBFDSurrogate(instance);
    }
    else if (algo_name == "BFD-ExtendedSum")
    {
        return new AlgoTSBFDExtendedSum(instance);
    }

    else if (algo_name == "FFD-DotProduct")
    {
        return new AlgoTSBinFFDDotProduct(instance);
    }
    else if (algo_name == "FFD-L2Norm")
    {
        return new AlgoTSBinFFDL2Norm(instance);
    }
    else if (algo_name == "FFD-Fitness")
    {
        return new AlgoTSBinFFDFitness(instance);
    }
    else
    {
        return nullptr; // This should never happen
    }
}

AlgoFitTS::AlgoFitTS(const InstanceTS & instance):
    instance_name(instance.getId()),
    size_TS(instance.getTSLength()),
    bin_cpu_capacity(instance.getBinCPUCapacity()),
    bin_mem_capacity(instance.getBinMemCapacity()),
    total_replicas(instance.getTotalReplicas()),
    sum_cpu_TS(instance.getSumCPUTS()),
    sum_mem_TS(instance.getSumMemTS()),
    next_bin_index(0),
    curr_bin_index(0),
    apps(AppListTS(instance.getApps())),
    bins(0),
    solved(false)
{ }

AlgoFitTS::~AlgoFitTS()
{
    for (BinTS* bin : bins)
    {
        if (bin != nullptr)
        {
            delete bin;
        }
    }
}

bool AlgoFitTS::isSolved() const
{
    return solved;
}

int AlgoFitTS::getSolution() const
{
    return bins.size();
}

const BinListTS& AlgoFitTS::getBins() const
{
    return bins;
}

BinListTS AlgoFitTS::getBinsCopy() const
{
    BinListTS new_bins;
    new_bins.reserve(bins.size());
    for (BinTS* bin : bins)
    {
        new_bins.push_back(new BinTS(*bin));
    }
    return new_bins;
}

const AppListTS& AlgoFitTS::getApps() const
{
    return apps;
}

const int AlgoFitTS::getBinCPUCapacity() const
{
    return bin_cpu_capacity;
}

const int AlgoFitTS::getBinMemCapacity() const
{
    return bin_mem_capacity;
}

const std::string& AlgoFitTS::getInstanceName() const
{
    return instance_name;
}

void AlgoFitTS::setSolution(BinListTS& bins)
{
    clearSolution();
    this->bins = bins;
    solved = true;
}

void AlgoFitTS::clearSolution()
{
    solved = false;
    for (BinTS* bin : bins)
    {
        if (bin != nullptr)
        {
            delete bin;
        }
    }
    bins.clear();
    next_bin_index = 0;
}

void AlgoFitTS::createNewBin()
{
    bins.push_back(new BinTS(next_bin_index, bin_cpu_capacity, bin_mem_capacity, size_TS));
    next_bin_index += 1;
}


// Generic algorithm based on first fit
void AlgoFitTS::allocateBatch(AppListTS::iterator first_app, AppListTS::iterator end_batch)
{
    BinTS* curr_bin = nullptr;
    bool allocated = false;

    sortApps(first_app, end_batch);
    auto curr_app_it = first_app;
    while(curr_app_it != end_batch)
    {
        ApplicationTS * app = *curr_app_it;
        curr_bin_index = 0;

        for (int j = 0; j < app->getNbReplicas(); ++j)
        {
            sortBins();

            allocated = false;
            while (!allocated)
            {
                if (curr_bin_index >= next_bin_index)
                {
                    // Create a new bin
                    createNewBin();

                    // This is a quick safe guard tailored for the TClab dataset to avoid infinite loops and running out of memory
                    if (bins.size() > total_replicas)
                    {
                        std::cout << "There seem to be a problem with instance " << getInstanceName() << ", created " << bins.size() << " bins" << std::endl; //TODO remove
                        return;
                    }
                }

                curr_bin = bins.at(curr_bin_index);
                if (checkItemToBin(app, curr_bin))
                {
                    // This depends whether to update conflicts/affinities of the bin
                    addItemToBin(app, j, curr_bin);
                    allocated = true;
                }
                else
                {
                    curr_bin_index += 1;
                }
            }
        }
        curr_app_it++;
    }
}


// The hint is an estimate on the number of bins to allocate
int AlgoFitTS::solveInstance(int hint_nb_bins)
{
    if(isSolved())
    {
        return getSolution(); // No need to solve twice
    }
    if(hint_nb_bins > 0)
    {
        bins.reserve(hint_nb_bins); // Small memory optimisation
    }

    allocateBatch(apps.begin(), apps.end());

    solved = true;
    return getSolution();
}

// Solve the instance per batches of batch_size apps at a time
// The hint is an estimate on the number of bins to allocate
int AlgoFitTS::solvePerBatch(int batch_size, int hint_nb_bins)
{
    if(isSolved())
    {
        return getSolution(); // No need to solve twice
    }
    if(hint_nb_bins > 0)
    {
        bins.reserve(hint_nb_bins);
    }

    int remaining_apps = apps.size();
    AppListTS::iterator curr_app_it;
    AppListTS::iterator stop_it = apps.begin();
    while(remaining_apps > batch_size)
    {
        curr_app_it = stop_it;
        stop_it = curr_app_it + batch_size;
        allocateBatch(curr_app_it, stop_it);

        // batch_size apps were allocated at this round
        remaining_apps -= batch_size;
    }

    // One last round with remaining apps (< batch_size)
    allocateBatch(stop_it, apps.end());

    solved = true;
    return getSolution();
}


/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ First Fit Affinity *********/
AlgoTSFF::AlgoTSFF(const InstanceTS &instance):
    AlgoFitTS(instance)
{ }

void AlgoTSFF::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it) { }
void AlgoTSFF::sortBins() { }

bool AlgoTSFF::checkItemToBin(ApplicationTS* app, BinTS* bin) const
{
    return (bin->doesItemFit(app)) and (bin->isAffinityCompliant(app));
}

void AlgoTSFF::addItemToBin(ApplicationTS* app, int replica_id, BinTS* bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);
}


/************ First Fit Decreasing Degree Affinity *********/
AlgoTSFFDDegree::AlgoTSFFDDegree(const InstanceTS &instance):
    AlgoTSFF(instance)
{ }

void AlgoTSFFDDegree::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_total_degree_decreasing);
}


/************ First Fit Decreasing Average Affinity *********/
AlgoTSFFDAvg::AlgoTSFFDAvg(const InstanceTS &instance):
    AlgoTSFF(instance)
{ }

void AlgoTSFFDAvg::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avg_size_decreasing);
}


/************ First Fit Decreasing Max Affinity *********/
AlgoTSFFDMax::AlgoTSFFDMax(const InstanceTS &instance):
    AlgoTSFF(instance)
{ }

void AlgoTSFFDMax::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_max_size_decreasing);
}


/************ FFD Average with Exponential Weights Affinity *********/
AlgoTSFFDAvgExpo::AlgoTSFFDAvgExpo(const InstanceTS &instance):
    AlgoTSFF(instance)
{ }

void AlgoTSFFDAvgExpo::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avgexpo_size_decreasing);
}


/************ First Fit Decreasing Surrogate Affinity *********/
AlgoTSFFDSurrogate::AlgoTSFFDSurrogate(const InstanceTS &instance):
    AlgoTSFF(instance)
{ }

void AlgoTSFFDSurrogate::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_surrogate_size_decreasing);
}


/************ First Fit Decreasing Extended Sum Affinity *********/
AlgoTSFFDExtendedSum::AlgoTSFFDExtendedSum(const InstanceTS &instance):
    AlgoTSFF(instance)
{ }

void AlgoTSFFDExtendedSum::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_extsum_size_decreasing);
}




/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ Best Fit Decreasing Average Affinity *********/
AlgoTSBFDAvg::AlgoTSBFDAvg(const InstanceTS &instance):
    AlgoFitTS(instance)
{ }

void AlgoTSBFDAvg::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avg_size_decreasing);
}

void AlgoTSBFDAvg::sortBins() {
    // The measure of the bins should have been updated before
    // Only need to sort the bins from current onward
    // Bins before curr_bin_index cannot accomodate replicas of the current application
    auto start_bin = bins.begin() + curr_bin_index;
    bubble_bin_up(start_bin, bins.end(), bin2D_comparator_measure_increasing);
}

bool AlgoTSBFDAvg::checkItemToBin(ApplicationTS* app, BinTS* bin) const
{
    return (bin->doesItemFit(app)) and (bin->isAffinityCompliant(app));
}

void AlgoTSBFDAvg::addItemToBin(ApplicationTS* app, int replica_id, BinTS* bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);

    updateBinMeasure(bin);
}

void AlgoTSBFDAvg::updateBinMeasure(BinTS *bin)
{
    // measure = normalised residual cpu + normalised residual memory
    float measure = (bin->getTotalResidualCPU() / bin->getMaxCPUCap()) + (bin->getTotalResidualMem() / bin->getMaxMemCap());
    bin->setMeasure(measure);
}


/************ Best Fit Decreasing Max Affinity *********/
AlgoTSBFDMax::AlgoTSBFDMax(const InstanceTS &instance):
    AlgoTSBFDAvg(instance)
{ }

void AlgoTSBFDMax::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_max_size_decreasing);
}

void AlgoTSBFDMax::updateBinMeasure(BinTS *bin)
{
    // measure = max(normalised residual cpu, normalised residual memory)
    const ResourceTS& bin_cpu_caps = bin->getAvailableCPUCaps();
    const ResourceTS& bin_mem_caps = bin->getAvailableMemCaps();
    float max_cpu = 0.0;
    float max_mem = 0.0;
    for (size_t i = 0; i < size_TS; ++i)
    {
        if ( bin_cpu_caps[i] > max_cpu)
        {
            max_cpu = bin_cpu_caps[i];
        }
        if (bin_mem_caps[i] > max_mem)
        {
            max_mem = bin_mem_caps[i];
        }
    }
    float measure = std::max((max_cpu / bin->getMaxCPUCap()), (max_mem / bin->getMaxMemCap()));
    bin->setMeasure(measure);
}



/************ Best Average with Exponential Weights Affinity *********/
AlgoTSBFDAvgExpo::AlgoTSBFDAvgExpo(const InstanceTS &instance):
    AlgoTSBFDAvg(instance),
    total_residual_cpu(0.0),
    total_residual_mem(0.0)
{ }

void AlgoTSBFDAvgExpo::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avgexpo_size_decreasing);
}

void AlgoTSBFDAvgExpo::createNewBin()
{
    bins.push_back(new BinTS(next_bin_index, bin_cpu_capacity, bin_mem_capacity, size_TS));
    next_bin_index += 1;

    total_residual_cpu += bin_cpu_capacity * size_TS;
    total_residual_mem += bin_mem_capacity * size_TS;
}

void AlgoTSBFDAvgExpo::addItemToBin(ApplicationTS *app, int replica_id, BinTS *bin)
{
    // To update only with the total capacity of the app
    total_residual_cpu -= bin->getTotalResidualCPU();
    total_residual_mem -= bin->getTotalResidualMem();

    bin->addNewConflict(app);
    bin->addItem(app, replica_id);

    // To update only with the total capacity of the app
    // In the end, only the total consumption of the app was removed
    total_residual_cpu += bin->getTotalResidualCPU();
    total_residual_mem += bin->getTotalResidualMem();
}


void AlgoTSBFDAvgExpo::updateBinMeasure(BinTS *bin)
{
    // measure = exp(0.01* (sum residual capacity all bins)/(nb bin * bin capacity) * normalised residual cpu + same with memory
    int bin_cpu_cap = bin->getMaxCPUCap();
    int bin_mem_cap = bin->getMaxMemCap();
    float factor_cpu = std::exp(0.01 * total_residual_cpu / (bin_cpu_cap * bins.size())) / bin_cpu_cap;
    float factor_mem = std::exp(0.01 * total_residual_mem / (bin_mem_cap * bins.size())) / bin_mem_cap;

    for(auto it_bin = (bins.begin()+curr_bin_index); it_bin != bins.end(); ++it_bin)
    {
        float measure = factor_cpu * (*it_bin)->getAvailableCPUCap() + factor_mem * (*it_bin)->getAvailableMemCap();
        (*it_bin)->setMeasure(measure);
    }
}



/************ Best Fit Decreasing Surrogate Affinity *********/
AlgoTSBFDSurrogate::AlgoTSBFDSurrogate(const InstanceTS &instance):
    AlgoTSBFDAvgExpo(instance)
{ }

void AlgoTSBFDSurrogate::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_surrogate_size_decreasing);
}

void AlgoTSBFDSurrogate::updateBinMeasure(BinTS *bin)
{
    // measure = lambda norm residual cpu + (1-lambda) * norm residual mem
    int bin_cpu_cap = bin->getMaxCPUCap();
    int bin_mem_cap = bin->getMaxMemCap();

    float lambda = total_residual_cpu / (total_residual_cpu + total_residual_mem);

    for(auto it_bin = (bins.begin()+curr_bin_index); it_bin != bins.end(); ++it_bin)
    {
        float measure = lambda * ((*it_bin)->getAvailableCPUCap() / bin_cpu_cap) + (1-lambda) * ((*it_bin)->getAvailableMemCap() / bin_mem_cap);
        (*it_bin)->setMeasure(measure);
    }
}




/************ Best Fit Decreasing Extended Sum Affinity *********/
AlgoTSBFDExtendedSum::AlgoTSBFDExtendedSum(const InstanceTS &instance):
    AlgoTSBFDAvgExpo(instance),
    sum_residual_cpu(size_TS, 0.0),
    sum_residual_mem(size_TS, 0.0)
{ }

void AlgoTSBFDExtendedSum::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_extsum_size_decreasing);
}

void AlgoTSBFDExtendedSum::createNewBin()
{
    bins.push_back(new BinTS(next_bin_index, bin_cpu_capacity, bin_mem_capacity, size_TS));
    next_bin_index += 1;

    for(size_t i = 0; i < size_TS; ++i)
    {
        sum_residual_cpu[i] += bin_cpu_capacity;
        sum_residual_mem[i] += bin_mem_capacity;
    }
}

void AlgoTSBFDExtendedSum::addItemToBin(ApplicationTS *app, int replica_id, BinTS *bin)
{
    // TODO the time complexity could be improved!
    // To update only with the total capacity of the app
    const ResourceTS& bin_cpu_caps = bin->getAvailableCPUCaps();
    const ResourceTS& bin_mem_caps = bin->getAvailableMemCaps();
    for(size_t i = 0; i < size_TS; ++i)
    {
        sum_residual_cpu[i] -= bin_cpu_caps[i];
        sum_residual_mem[i] -= bin_mem_caps[i];
    }

    bin->addNewConflict(app);
    bin->addItem(app, replica_id);

    // To update only with the total capacity of the app
    // In the end, only the total consumption of the app was removed
    for(size_t i = 0; i < size_TS; ++i)
    {
        sum_residual_cpu[i] -= bin_cpu_caps[i];
        sum_residual_mem[i] -= bin_mem_caps[i];
    }
}

void AlgoTSBFDExtendedSum::updateBinMeasure(BinTS *bin)
{
    // measure = residual cpu / total residual cpu + residual mem / total residual mem
    // for each time step
    // (no need to use normalised values here)
    // Compute the sum of residual capacity for each time step
    for (auto it_bin = (bins.begin()+curr_bin_index); it_bin != bins.end(); ++it_bin)
    {
        const ResourceTS& bin_cpu_caps = (*it_bin)->getAvailableCPUCaps();
        const ResourceTS& bin_mem_caps = (*it_bin)->getAvailableMemCaps();

        //float measure = ((float)b->getAvailableCPUCap()) / total_residual_cpu + ((float)b->getAvailableMemCap()) / total_residual_mem;
        float measure = 0.0;
        for (size_t i = 0; i < size_TS; ++i)
        {
            measure += bin_cpu_caps[i] / sum_residual_cpu[i];
            measure += bin_mem_caps[i] / sum_residual_mem[i];
        }
        (*it_bin)->setMeasure(measure);
    }
}




/* ================================================ */
/* ================================================ */
/* ================================================ */
/********* Bin Centric FFD DotProduct ***************/
AlgoTSBinFFDDotProduct::AlgoTSBinFFDDotProduct(const InstanceTS &instance):
    AlgoTSFF(instance)
{ }

bool AlgoTSBinFFDDotProduct::isBinFilled(BinTS* bin)
{
    const ResourceTS& bin_cpu_caps = bin->getAvailableCPUCaps();
    const ResourceTS& bin_mem_caps = bin->getAvailableMemCaps();
    for (size_t i = 0; i < size_TS; ++i)
    {
        if ((bin_cpu_caps[i] <= 0.0) or (bin_mem_caps[i] <= 0.0))
        {
            return true;
        }
    }
    return false;
}

void AlgoTSBinFFDDotProduct::computeMeasures(AppListTS::iterator start_list, AppListTS::iterator end_list, BinTS *bin)
{
    const ResourceTS& bin_cpu_caps = bin->getAvailableCPUCaps();
    const ResourceTS& bin_mem_caps = bin->getAvailableMemCaps();
    for(auto it = start_list; it != end_list; ++it)
    {
        ApplicationTS * app = *it;
        // Use normalized values of app size and bin residual capacity
        const ResourceTS& app_norm_cpu = app->getNormCpuUsage();
        const ResourceTS& app_norm_mem = app->getNormMemUsage();
        float measure = 0.0;
        for (size_t i = 0; i < size_TS; ++i)
        {
            measure += app_norm_cpu[i] * bin_cpu_caps[i] / bin->getMaxCPUCap();
            measure += app_norm_mem[i] * bin_mem_caps[i] / bin->getMaxMemCap();
        }

        app->setMeasure(measure);
    }
}

void AlgoTSBinFFDDotProduct::allocateBatch(AppListTS::iterator first_app, AppListTS::iterator end_batch)
{
    std::unordered_map<std::string, int> next_id_replicas;
    next_id_replicas.reserve((end_batch - first_app)); // Stores the id of the next replica to pack for each application

    for(auto it = first_app; it != end_batch; ++it)
    {
        next_id_replicas[(*it)->getId()] = 0;
    }

    BinTS* curr_bin = nullptr;

    // While there are still applications to pack
    auto next_treated_app_it = first_app;
    auto end_list = end_batch;
    while(next_treated_app_it != end_list)
    {
        // Open a new bin
        curr_bin = new BinTS(next_bin_index, bin_cpu_capacity, bin_mem_capacity, size_TS);
        next_bin_index += 1;
        bins.push_back(curr_bin);

        auto current_app_it = next_treated_app_it;
        // This is a quick safe guard to avoid infinite loops and running out of memory
        if (bins.size() > total_replicas)
        {
            std::cout << "There seem to be a problem with instance " << getInstanceName() << ", created " << bins.size() << " bins" << std::endl; //TODO remove
            return;
        }

        // Fill the bin as much as possible
        bool continue_loop = true;
        while(continue_loop and (current_app_it != end_list))
        {
            // Re-order the list following the measure
            computeMeasures(current_app_it, end_list, curr_bin);
            bubble_appsTS_up(current_app_it, end_list, application2D_comparator_measure_decreasing);

            // Retrieve the next application to pack
            ApplicationTS* app = *current_app_it;

            // Try to pack as much replicas as possible
            int replica_id = next_id_replicas[app->getId()];

            bool could_pack = true;
            while ( (replica_id < app->getNbReplicas()) and could_pack)
            {
                if (checkItemToBin(app, curr_bin))
                {
                    addItemToBin(app, replica_id, curr_bin);
                    replica_id +=1;
                }
                else
                {
                    could_pack = false;
                }
            }

            next_id_replicas[app->getId()] = replica_id;

            // If no more replicas to pack, put the app in the fully packed zone
            if (replica_id >= app->getNbReplicas())
            {
                std::iter_swap(next_treated_app_it, current_app_it);
                next_treated_app_it++;
            }

            current_app_it++;

            // If the bin is filled, stop the loop
            if (isBinFilled(curr_bin))
            {
                continue_loop = false;
            }
        }
    }
}



/********* Bin Centric FFD L2Norm ***************/
AlgoTSBinFFDL2Norm::AlgoTSBinFFDL2Norm(const InstanceTS &instance):
    AlgoTSBinFFDDotProduct(instance)
{ }

void AlgoTSBinFFDL2Norm::computeMeasures(AppListTS::iterator start_list, AppListTS::iterator end_list, BinTS *bin)
{
    const ResourceTS& bin_cpu_caps = bin->getAvailableCPUCaps();
    const ResourceTS& bin_mem_caps = bin->getAvailableMemCaps();
    for(auto it = start_list; it != end_list; ++it)
    {
        ApplicationTS * app = *it;
        // Use normalized values of app size and bin residual capacity
        const ResourceTS& app_norm_cpu = app->getNormCpuUsage();
        const ResourceTS& app_norm_mem = app->getNormMemUsage();
        float measure = 0.0;
        for (size_t i = 0; i < size_TS; ++i)
        {
            float a = (bin_cpu_caps[i] / bin->getMaxCPUCap()) - app_norm_cpu[i];
            float b = (bin_mem_caps[i] / bin->getMaxMemCap()) - app_norm_mem[i];
            measure += a*a + b*b;
        }

        app->setMeasure(measure);
    }
}



/********* Bin Centric FFD Fitness ***************/
AlgoTSBinFFDFitness::AlgoTSBinFFDFitness(const InstanceTS &instance):
    AlgoTSBinFFDDotProduct(instance)
{ }

void AlgoTSBinFFDFitness::computeMeasures(AppListTS::iterator start_list, AppListTS::iterator end_list, BinTS *bin)
{
    // TODO This complexity is too much
    // Could be greatly improved by keeping the values of the already
    // packed bins (since the algo is bin-centric)
    // Compute the resudual capacity of bins for each point of the time series
    ResourceTS sum_res_cpu(size_TS, 0.0);
    ResourceTS sum_res_mem(size_TS, 0.0);
    for(BinTS * bb : bins)
    {
        const ResourceTS& bin_res_cpu = bb->getAvailableCPUCaps();
        const ResourceTS& bin_res_mem = bb->getAvailableMemCaps();

        for (size_t i = 0; i < size_TS; ++i)
        {
            sum_res_cpu[i] += bin_res_cpu[i];
            sum_res_mem[i] += bin_res_mem[i];
        }
    }
    for (size_t i = 0; i < size_TS; ++i)
    {
        sum_res_cpu[i] = sum_res_cpu[i] / bin->getMaxCPUCap();
        sum_res_mem[i] = sum_res_mem[i] / bin->getMaxMemCap();
    }
    // Use normalized values of app size and bin residual capacity
    const ResourceTS& bin_res_cpu = bin->getAvailableCPUCaps();
    const ResourceTS& bin_res_mem = bin->getAvailableMemCaps();
    for(auto it = start_list; it != end_list; ++it)
    {
        ApplicationTS * app = *it;
        // Use normalized values of app size and bin residual capacity
        const ResourceTS& app_norm_cpu = app->getNormCpuUsage();
        const ResourceTS& app_norm_mem = app->getNormMemUsage();
        float measure = 0.0;
        for (size_t i = 0; i < size_TS; ++i)
        {
            float a = (app_norm_cpu[i] * bin_res_cpu[i]) / (sum_cpu_TS[i] * sum_res_cpu[i] * bin->getMaxCPUCap());
            float b = (app_norm_mem[i] * bin_res_mem[i]) / (sum_mem_TS[i] * sum_res_mem[i] * bin->getMaxMemCap());
            measure += a + b;
        }

        app->setMeasure(measure);
    }
}

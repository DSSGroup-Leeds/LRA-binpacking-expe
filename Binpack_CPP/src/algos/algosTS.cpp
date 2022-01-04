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

    else if(algo_name == "WFD-Avg")
    {
        return new AlgoTSWFDAvg(instance);
    }
    else if(algo_name == "WFD-Max")
    {
        return new AlgoTSWFDMax(instance);
    }
    else if(algo_name == "WFD-AvgExpo")
    {
        return new AlgoTSWFDAvgExpo(instance);
    }
    else if(algo_name == "WFD-Surrogate")
    {
        return new AlgoTSWFDSurrogate(instance);
    }
    else if (algo_name == "WFD-ExtendedSum")
    {
        return new AlgoTSWFDExtendedSum(instance);
    }

    else if (algo_name == "NCD-DotProduct")
    {
        return new AlgoTSBinFFDDotProduct(instance);
    }
    else if (algo_name == "NCD-DotDivision")
    {
        return new AlgoTSBinFFDDotDivision(instance);
    }
    else if (algo_name == "NCD-L2Norm")
    {
        return new AlgoTSBinFFDL2Norm(instance);
    }
    else if (algo_name == "NCD-Fitness")
    {
        return new AlgoTSBinFFDFitness(instance);
    }
    else
    {
        return nullptr; // This should never happen
    }
}

AlgoTSSpreadWFDAvg* createSpreadAlgo(const std::string &algo_name, const InstanceTS &instance)
{
    if (algo_name == "SpreadWFD-Avg")
    {
        return new AlgoTSSpreadWFDAvg(instance);
    }
    else if (algo_name == "SpreadWFD-Max")
    {
        return new AlgoTSSpreadWFDMax(instance);
    }
    else if (algo_name == "SpreadWFD-Surrogate")
    {
        return new AlgoTSSpreadWFDSurrogate(instance);
    }
    /*else if (algo_name == "SpreadWFD-AvgExpo")
    {
        return new AlgoTSSpreadWFDAvgExpo(instance);
    }*/
    /*else if (algo_name == "SpreadWFD-ExtendedSum")
    {
        return new AlgoTSSpreadWFDExtendedSum(instance);
    }*/
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
    bubble_bin_up(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_increasing);
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
    float measure = (bin->getTotalResidualCPU() / bin_cpu_capacity) + (bin->getTotalResidualMem() / bin_mem_capacity);
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
    float measure = std::max((max_cpu / bin_cpu_capacity), (max_mem / bin_mem_capacity));
    bin->setMeasure(measure);
}



/************ Best Average with Exponential Weights Affinity *********/
AlgoTSBFDAvgExpo::AlgoTSBFDAvgExpo(const InstanceTS &instance):
    AlgoTSBFDAvg(instance),
    sum_residual_cpu(size_TS, 0.0),
    sum_residual_mem(size_TS, 0.0)
{ }

void AlgoTSBFDAvgExpo::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avgexpo_size_decreasing);
}

void AlgoTSBFDAvgExpo::sortBins() {
    // The measure of the bins should have been updated before
    // Need to sort all bins since all measures have changed
    stable_sort(bins.begin()+curr_bin_index, bins.end(), bin2D_comparator_measure_increasing);
}

void AlgoTSBFDAvgExpo::createNewBin()
{
    bins.push_back(new BinTS(next_bin_index, bin_cpu_capacity, bin_mem_capacity, size_TS));
    next_bin_index += 1;

    for(size_t i = 0; i < size_TS; ++i)
    {
        sum_residual_cpu[i] += bin_cpu_capacity;
        sum_residual_mem[i] += bin_mem_capacity;
    }
}

void AlgoTSBFDAvgExpo::addItemToBin(ApplicationTS *app, int replica_id, BinTS *bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);

    const ResourceTS& app_cpu = app->getCpuUsage();
    const ResourceTS& app_mem = app->getMemUsage();

    for(size_t i = 0; i < size_TS; ++i)
    {
        sum_residual_cpu[i] -= app_cpu[i];
        sum_residual_mem[i] -= app_mem[i];
    }

    updateBinMeasure(bin);
}


void AlgoTSBFDAvgExpo::updateBinMeasure(BinTS *bin)
{
    ResourceTS factors_cpu(size_TS, 0.0);
    ResourceTS factors_mem(size_TS, 0.0);

    for(size_t i = 0; i < size_TS; ++i)
    {
        factors_cpu[i] = std::exp(0.01 * sum_residual_cpu[i] / (bin_cpu_capacity * bins.size())) / bin_cpu_capacity;
        factors_mem[i] = std::exp(0.01 * sum_residual_mem[i] / (bin_mem_capacity * bins.size())) / bin_mem_capacity;
    }

    for(auto it_bin = (bins.begin()+curr_bin_index); it_bin != bins.end(); ++it_bin)
    {
        const ResourceTS& bin_cpu_caps = (*it_bin)->getAvailableCPUCaps();
        const ResourceTS& bin_mem_caps = (*it_bin)->getAvailableMemCaps();
        float measure = 0.0;
        for(size_t i = 0; i < size_TS; ++i)
        {
            // For all timestep and cpu/memory
            // measure += exp(0.01 * (sum residual capacity all bins) / (nb bins * bin capacity)) * norm residual capacity
            // No need to normalized bin residual capaciies here because already done in factors
            measure += factors_cpu[i] * bin_cpu_caps[i] + factors_mem[i] * bin_mem_caps[i];
        }
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

void AlgoTSBFDSurrogate::sortBins() {
    // The measure of the bins should have been updated before
    // Need to sort all bins since all measures have changed
    stable_sort(bins.begin()+curr_bin_index, bins.end(), bin2D_comparator_measure_increasing);
}

void AlgoTSBFDSurrogate::updateBinMeasure(BinTS *bin)
{
    // measure = lambda norm residual cpu + (1-lambda) * norm residual mem
    // adapted for time series
    float lambda_ratio = 0.0;
    for(size_t i = 0; i < size_TS; ++i)
    {
        // Compute sum of normalzed residual capacities
        lambda_ratio += sum_residual_cpu[i] + sum_residual_mem[i];
    }

    for(auto it_bin = (bins.begin()+curr_bin_index); it_bin != bins.end(); ++it_bin)
    {
        const ResourceTS& bin_cpu_caps = (*it_bin)->getAvailableCPUCaps();
        const ResourceTS& bin_mem_caps = (*it_bin)->getAvailableMemCaps();
        float measure = 0.0;
        for(size_t i = 0; i < size_TS; ++i)
        {
            measure += (sum_residual_cpu[i] / lambda_ratio) * bin_cpu_caps[i] / bin_cpu_capacity;
            measure += (sum_residual_mem[i] / lambda_ratio) * bin_mem_caps[i] / bin_mem_capacity;
        }
        (*it_bin)->setMeasure(measure);
    }
}




/************ Best Fit Decreasing Extended Sum Affinity *********/
AlgoTSBFDExtendedSum::AlgoTSBFDExtendedSum(const InstanceTS &instance):
    AlgoTSBFDAvgExpo(instance)
{ }

void AlgoTSBFDExtendedSum::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_extsum_size_decreasing);
}

void AlgoTSBFDExtendedSum::sortBins() {
    // The measure of the bins should have been updated before
    // Need to sort all bins since all measures have changed
    stable_sort(bins.begin()+curr_bin_index, bins.end(), bin2D_comparator_measure_increasing);
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
/************ Worst Fit Decreasing Avg Affinity *********/
AlgoTSWFDAvg::AlgoTSWFDAvg(const InstanceTS &instance):
    AlgoTSBFDAvg(instance)
{ }

void AlgoTSWFDAvg::sortBins() {
    bubble_bin_up(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
}

/************ Worst Fit Decreasing Max Affinity *********/
AlgoTSWFDMax::AlgoTSWFDMax(const InstanceTS &instance):
    AlgoTSBFDMax(instance)
{ }

void AlgoTSWFDMax::sortBins() {
    bubble_bin_up(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
}


/************ Worst Fit Decreasing AvgExpo Affinity *********/
AlgoTSWFDAvgExpo::AlgoTSWFDAvgExpo(const InstanceTS &instance):
    AlgoTSBFDAvgExpo(instance)
{ }

void AlgoTSWFDAvgExpo::sortBins() {
    stable_sort(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
}


/************ Worst Fit Decreasing Surrogate Affinity *********/
AlgoTSWFDSurrogate::AlgoTSWFDSurrogate(const InstanceTS &instance):
    AlgoTSBFDSurrogate(instance)
{ }

void AlgoTSWFDSurrogate::sortBins() {
    stable_sort(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
}


/************ Worst Fit Decreasing ExtendedSum Affinity *********/
AlgoTSWFDExtendedSum::AlgoTSWFDExtendedSum(const InstanceTS &instance):
    AlgoTSBFDExtendedSum(instance)
{ }

void AlgoTSWFDExtendedSum::sortBins() {
    stable_sort(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
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
            measure += app_norm_cpu[i] * bin_cpu_caps[i] / bin_cpu_capacity;
            measure += app_norm_mem[i] * bin_mem_caps[i] / bin_mem_capacity;
        }

        app->setMeasure(measure);
    }
}

BinTS* AlgoTSBinFFDDotProduct::createNewBinRet()
{
    BinTS* bin = new BinTS(next_bin_index, bin_cpu_capacity, bin_mem_capacity, size_TS);
    next_bin_index += 1;
    bins.push_back(bin);
    return bin;
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
        curr_bin = createNewBinRet();

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



/********* Bin Centric FFD DotDivision ***************/
AlgoTSBinFFDDotDivision::AlgoTSBinFFDDotDivision(const InstanceTS &instance):
    AlgoTSBinFFDDotProduct(instance)
{ }

void AlgoTSBinFFDDotDivision::computeMeasures(AppListTS::iterator start_list, AppListTS::iterator end_list, BinTS *bin)
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
            measure += app_norm_cpu[i] * bin_cpu_capacity / bin_cpu_caps[i];
            measure += app_norm_mem[i] * bin_mem_capacity / bin_mem_caps[i];
        }

        app->setMeasure(measure);
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
            float a = (bin_cpu_caps[i] / bin_cpu_capacity) - app_norm_cpu[i];
            float b = (bin_mem_caps[i] / bin_mem_capacity) - app_norm_mem[i];
            measure += a*a + b*b;
        }

        // Minus sign to have reverse order
        app->setMeasure(-measure);
    }
}



/********* Bin Centric FFD Fitness ***************/
AlgoTSBinFFDFitness::AlgoTSBinFFDFitness(const InstanceTS &instance):
    AlgoTSBinFFDDotProduct(instance),
    sum_residual_cpu(size_TS, 0.0),
    sum_residual_mem(size_TS, 0.0)
{ }

BinTS* AlgoTSBinFFDFitness::createNewBinRet()
{
    BinTS* bin = AlgoTSBinFFDDotProduct::createNewBinRet();

    for (size_t i = 0; i < size_TS; ++i)
    {
        sum_residual_cpu[i] += bin_cpu_capacity;
        sum_residual_mem[i] += bin_mem_capacity;
    }

    return bin;
}

void AlgoTSBinFFDFitness::addItemToBin(ApplicationTS *app, int replica_id, BinTS *bin)
{
    AlgoTSBinFFDDotProduct::addItemToBin(app, replica_id, bin);

    const ResourceTS& app_cpu = app->getCpuUsage();
    const ResourceTS& app_mem = app->getMemUsage();

    for (size_t i = 0; i < size_TS; ++i)
    {
        sum_residual_cpu[i] -= app_cpu[i];
        sum_residual_mem[i] -= app_mem[i];
    }
}

void AlgoTSBinFFDFitness::computeMeasures(AppListTS::iterator start_list, AppListTS::iterator end_list, BinTS *bin)
{
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
            // No need to use normalized values of total and residual bin capacities
            float a = (app_norm_cpu[i] * bin_res_cpu[i]) / (sum_cpu_TS[i] * sum_residual_cpu[i]);
            float b = (app_norm_mem[i] * bin_res_mem[i]) / (sum_mem_TS[i] * sum_residual_mem[i]);
            measure += a + b;
        }

        app->setMeasure(measure);
    }
}



/* ================================================ */
/* ================================================ */
/* ================================================ */
/*********** Spread replicas Worst Fit Avg **********/
AlgoTSSpreadWFDAvg::AlgoTSSpreadWFDAvg(const InstanceTS &instance):
    AlgoFitTS(instance)
{ }

int AlgoTSSpreadWFDAvg::solveInstanceSpread(int LB_bins, int UB_bins)
{
    // First, try to find a solution with UB_bins
    if (!trySolve(UB_bins))
    {
        // If no solution found, refine values of LB and increase UB
        bool sol_found = false;
        while (!sol_found)
        {
            LB_bins = UB_bins+1; // No solution was found, UB+1 is a new lower bound
            UB_bins += 51; // +1 to be in par with the +1 of LB

            sol_found = trySolve(UB_bins);
        }
    }

    // Store the current solution
    BinListTS best_bins = getBinsCopy();
    int best_sol = UB_bins;
    int low_bound = LB_bins;
    int target_bins;

    // Then iteratively try to improve on the solution
    while(low_bound < best_sol)
    {
        target_bins = (low_bound + best_sol)/2;

        if (trySolve(target_bins))
        {
            // Update the best solution
            best_sol = target_bins;
            for (BinTS* bin : best_bins)
            {
                if (bin != nullptr)
                {
                    delete bin;
                }
            }
            best_bins.clear();
            best_bins = getBinsCopy();
        }
        else
        {
            // Update bound of search
            low_bound = target_bins+1;
        }
    }
    setSolution(best_bins);
    return best_sol;
}

bool AlgoTSSpreadWFDAvg::trySolve(int nb_bins)
{
    clearSolution();
    createBins(nb_bins);

    // For each app in the list, try to put all replicas in separate bins
    BinTS* curr_bin = nullptr;
    sortApps(apps.begin(), apps.end());
    auto current_app_it = apps.begin();
    while(current_app_it != apps.end())
    {
        ApplicationTS * app = *current_app_it;
        curr_bin_index = 0;
        int replica_index = app->getNbReplicas()-1;
        while(replica_index >= 0) // There are still replicas to pack
        {
            bool replica_packed = false;
            int start_bin_index = curr_bin_index;
            while(!replica_packed)
            {
                curr_bin = bins.at(curr_bin_index);
                if (checkItemToBin(app, curr_bin))
                {
                    addItemToBin(app, replica_index, curr_bin);
                    updateBinMeasure(curr_bin);
                    replica_packed = true;
                    replica_index -= 1;
                }

                // Advance to next bin
                if (curr_bin_index == (nb_bins-1))
                {
                    curr_bin_index = 0; // Reset
                }
                else
                {
                    curr_bin_index += 1;
                }

                // If all bins were checked and the item cannot be packed
                if (!replica_packed and (curr_bin_index == start_bin_index))
                {
                    //std::cout << "Could not pack replica " << replica_index << " of app " << app->getId() << std::endl;
                    return false;
                }
            }
        }
        //std::cout << "All replicas of app " << app->getId() << " were packed. Updating bins order" << std::endl;
        current_app_it++;
        updateBinMeasures();
        sortBins();
    }
    return true;
}

void AlgoTSSpreadWFDAvg::createBins(int nb_bins)
{
    bins.reserve(nb_bins);
    for (int i = 0; i < nb_bins; ++i)
    {
        BinTS* bin = new BinTS(i, bin_cpu_capacity, bin_mem_capacity, size_TS);
        updateBinMeasure(bin);
        bins.push_back(bin);
    }
}


void AlgoTSSpreadWFDAvg::updateBinMeasure(BinTS* bin)
{
    float measure = (bin->getTotalResidualCPU() / bin_cpu_capacity) + (bin->getTotalResidualMem() / bin_mem_capacity);
    bin->setMeasure(measure);
}

void AlgoTSSpreadWFDAvg::updateBinMeasures(){ }


void AlgoTSSpreadWFDAvg::allocateBatch(AppListTS::iterator first_app, AppListTS::iterator end_batch)
{
    std::cout << "For Spread algorithm please call 'solveInstanceSpread' instead" << std::endl;
    return;
}

void AlgoTSSpreadWFDAvg::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avg_size_decreasing);
}

void AlgoTSSpreadWFDAvg::sortBins()
{
    stable_sort(bins.begin(), bins.end(), bin2D_comparator_measure_decreasing);
}

bool AlgoTSSpreadWFDAvg::checkItemToBin(ApplicationTS* app, BinTS* bin) const
{
    return (bin->doesItemFit(app)) and (bin->isAffinityCompliant(app));
}

void AlgoTSSpreadWFDAvg::addItemToBin(ApplicationTS* app, int replica_id, BinTS* bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);
}



/*********** Spread replicas Worst Fit Max **********/
AlgoTSSpreadWFDMax::AlgoTSSpreadWFDMax(const InstanceTS &instance):
    AlgoTSSpreadWFDAvg(instance)
{ }

void AlgoTSSpreadWFDMax::updateBinMeasure(BinTS* bin)
{
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
    float measure = std::max((max_cpu / bin_cpu_capacity), (max_mem / bin_mem_capacity));
    bin->setMeasure(measure);
}

void AlgoTSSpreadWFDMax::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_max_size_decreasing);
}




/*********** Spread replicas Worst Fit Surrogate **********/
AlgoTSSpreadWFDSurrogate::AlgoTSSpreadWFDSurrogate(const InstanceTS &instance):
    AlgoTSSpreadWFDAvg(instance),
    sum_residual_cpu(size_TS, 0.0),
    sum_residual_mem(size_TS, 0.0)
{ }

void AlgoTSSpreadWFDSurrogate::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_surrogate_size_decreasing);
}

void AlgoTSSpreadWFDSurrogate::updateBinMeasure(BinTS* bin) { }

void AlgoTSSpreadWFDSurrogate::updateBinMeasures()
{
    // measure = lambda norm residual cpu + (1-lambda) * norm residual mem
    float lambda_ratio = 0.0;
    for(size_t i = 0; i < size_TS; ++i)
    {
        // Compute sum of normalzed residual capacities
        lambda_ratio += sum_residual_cpu[i] + sum_residual_mem[i];
    }

    for(auto it_bin = (bins.begin()+curr_bin_index); it_bin != bins.end(); ++it_bin)
    {
        const ResourceTS& bin_cpu_caps = (*it_bin)->getAvailableCPUCaps();
        const ResourceTS& bin_mem_caps = (*it_bin)->getAvailableMemCaps();
        float measure = 0.0;
        for(size_t i = 0; i < size_TS; ++i)
        {
            measure += (sum_residual_cpu[i] / lambda_ratio) * bin_cpu_caps[i] / bin_cpu_capacity;
            measure += (sum_residual_mem[i] / lambda_ratio) * bin_mem_caps[i] / bin_mem_capacity;
        }
        (*it_bin)->setMeasure(measure);
    }
}

void AlgoTSSpreadWFDSurrogate::createBins(int nb_bins)
{
    bins.reserve(nb_bins);
    for (int i = 0; i < nb_bins; ++i)
    {
        BinTS* bin = new BinTS(i, bin_cpu_capacity, bin_mem_capacity, size_TS);
        bins.push_back(bin);
    }

    for(size_t i = 0; i < size_TS; ++i)
    {
        sum_residual_cpu[i] = nb_bins * bin_cpu_capacity;
        sum_residual_mem[i] = nb_bins * bin_mem_capacity;
    }
}

void AlgoTSSpreadWFDSurrogate::addItemToBin(ApplicationTS* app, int replica_id, BinTS* bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);

    const ResourceTS& app_cpu = app->getCpuUsage();
    const ResourceTS& app_mem = app->getMemUsage();

    for(size_t i = 0; i < size_TS; ++i)
    {
        sum_residual_cpu[i] -= app_cpu[i];
        sum_residual_mem[i] -= app_mem[i];
    }
}


/*********** Spread replicas Worst Fit AvgExpo **********/
/*AlgoTSSpreadWFDAvgExpo::AlgoTSSpreadWFDAvgExpo(const InstanceTS &instance):
    AlgoTSSpreadWFDAvg(instance)
{ }

void AlgoTSSpreadWFDAvgExpo::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avgexpo_size_decreasing);
}

void AlgoTSSpreadWFDAvgExpo::updateBinMeasure(BinTS* bin) { }

void AlgoTSSpreadWFDAvgExpo::updateBinMeasures()
{
    ResourceTS factors_cpu(size_TS, 0.0);
    ResourceTS factors_mem(size_TS, 0.0);

    for(size_t i = 0; i < size_TS; ++i)
    {
        factors_cpu[i] = std::exp(0.01 * sum_residual_cpu[i] / (bin_cpu_capacity * bins.size())) / bin_cpu_capacity;
        factors_mem[i] = std::exp(0.01 * sum_residual_mem[i] / (bin_mem_capacity * bins.size())) / bin_mem_capacity;
    }

    for(auto it_bin = (bins.begin()+curr_bin_index); it_bin != bins.end(); ++it_bin)
    {
        const ResourceTS& bin_cpu_caps = (*it_bin)->getAvailableCPUCaps();
        const ResourceTS& bin_mem_caps = (*it_bin)->getAvailableMemCaps();
        float measure = 0.0;
        for(size_t i = 0; i < size_TS; ++i)
        {
            // For all timestep and cpu/memory
            // measure += exp(0.01 * (sum residual capacity all bins) / (nb bins * bin capacity)) * norm residual capacity
            // No need to normalized bin residual capaciies here because already done in factors
            measure += factors_cpu[i] * bin_cpu_caps[i] + factors_mem[i] * bin_mem_caps[i];
        }
        (*it_bin)->setMeasure(measure);
    }
}*/



/*********** Spread replicas Worst Fit Extended Sum **********/
/*AlgoTSSpreadWFDExtendedSum::AlgoTSSpreadWFDExtendedSum(const InstanceTS &instance):
    AlgoTSSpreadWFDAvgExpo(instance)
{ }

void AlgoTSSpreadWFDExtendedSum::sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_extsum_size_decreasing);
}

void AlgoTSSpreadWFDExtendedSum::updateBinMeasure(BinTS* bin) { }

void AlgoTSSpreadWFDExtendedSum::updateBinMeasures()
{
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
}*/



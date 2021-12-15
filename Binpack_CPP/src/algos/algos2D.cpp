#include "algos2D.hpp"

#include <algorithm> // For stable_sort
#include <iostream>
#include <unordered_map>
#include <cmath> // For exp

AlgoFit2D* createAlgo2D(const std::string &algo_name, const Instance2D &instance)
{
    if (algo_name == "FF")
    {
        return new Algo2DFF(instance);
    }
    else if (algo_name == "FFD-Degree")
    {
        return new Algo2DFFDDegree(instance);
    }

    else if (algo_name == "FFD-Avg")
    {
        return new Algo2DFFDAvg(instance);
    }
    else if (algo_name == "FFD-Max")
    {
        return new Algo2DFFDMax(instance);
    }
    else if (algo_name == "FFD-AvgExpo")
    {
        return new Algo2DFFDAvgExpo(instance);
    }
    else if (algo_name == "FFD-Surrogate")
    {
        return new Algo2DFFDSurrogate(instance);
    }
    else if (algo_name == "FFD-ExtendedSum")
    {
        return new Algo2DFFDExtendedSum(instance);
    }

    else if(algo_name == "BFD-Avg")
    {
        return new Algo2DBFDAvg(instance);
    }
    else if(algo_name == "BFD-Max")
    {
        return new Algo2DBFDMax(instance);
    }
    else if(algo_name == "BFD-AvgExpo")
    {
        return new Algo2DBFDAvgExpo(instance);
    }
    else if(algo_name == "BFD-Surrogate")
    {
        return new Algo2DBFDSurrogate(instance);
    }
    else if(algo_name == "BFD-ExtendedSum")
    {
        return new Algo2DBFDExtendedSum(instance);
    }

    else if(algo_name == "WFD-Avg")
    {
        return new Algo2DWFDAvg(instance);
    }
    else if(algo_name == "WFD-Max")
    {
        return new Algo2DWFDMax(instance);
    }
    else if(algo_name == "WFD-AvgExpo")
    {
        return new Algo2DWFDAvgExpo(instance);
    }
    else if(algo_name == "WFD-Surrogate")
    {
        return new Algo2DWFDSurrogate(instance);
    }
    else if(algo_name == "WFD-ExtendedSum")
    {
        return new Algo2DWFDExtendedSum(instance);
    }

    else if(algo_name == "NodeCount")
    {
        return new Algo2DNodeCount(instance);
    }

    else if (algo_name == "NCD-DotProduct")
    {
        return new Algo2DBinFFDDotProduct(instance);
    }
    else if (algo_name == "NCD-DotDivision")
    {
        return new Algo2DBinFFDDotDivision(instance);
    }
    else if (algo_name == "NCD-L2Norm")
    {
        return new Algo2DBinFFDL2Norm(instance);
    }
    else if (algo_name == "NCD-Fitness")
    {
        return new Algo2DBinFFDFitness(instance);
    }
    else
    {
        return nullptr; // This should never happen
    }
}

Algo2DSpreadWFDAvg* createSpreadAlgo(const std::string &algo_name, const Instance2D &instance)
{
    if (algo_name == "SpreadWFD-Avg")
    {
        return new Algo2DSpreadWFDAvg(instance);
    }
    else if (algo_name == "SpreadWFD-Max")
    {
        return new Algo2DSpreadWFDMax(instance);
    }
    else if (algo_name == "SpreadWFD-AvgExpo")
    {
        return new Algo2DSpreadWFDAvgExpo(instance);
    }
    else if (algo_name == "SpreadWFD-Surrogate")
    {
        return new Algo2DSpreadWFDSurrogate(instance);
    }
    else if (algo_name == "SpreadWFD-ExtendedSum")
    {
        return new Algo2DSpreadWFDExtendedSum(instance);
    }
    else
    {
        return nullptr; // This should never happen
    }
}


AlgoFit2D::AlgoFit2D(const Instance2D &instance):
    instance_name(instance.getId()),
    bin_cpu_capacity(instance.getBinCPUCapacity()),
    bin_mem_capacity(instance.getBinMemCapacity()),
    total_replicas(instance.getTotalReplicas()),
    sum_cpu(instance.getSumCPU()),
    sum_mem(instance.getSumMem()),
    norm_sum_cpu(sum_cpu/bin_cpu_capacity),
    norm_sum_mem(sum_mem/bin_mem_capacity),
    next_bin_index(0),
    curr_bin_index(0),
    solved(false)
{
    apps = AppList2D(instance.getApps());
    bins = BinList2D(0);
}

AlgoFit2D::~AlgoFit2D()
{
    for (Bin2D* bin : bins)
    {
        if (bin != nullptr)
        {
            delete bin;
        }
    }
}

bool AlgoFit2D::isSolved() const
{
    return solved;
}

int AlgoFit2D::getSolution() const
{
    return bins.size();
}

const BinList2D& AlgoFit2D::getBins() const
{
    return bins;
}

BinList2D AlgoFit2D::getBinsCopy() const
{
    BinList2D new_bins;
    new_bins.reserve(bins.size());
    for (Bin2D* bin : bins)
    {
        new_bins.push_back(new Bin2D(*bin));
    }
    return new_bins;
}

const AppList2D& AlgoFit2D::getApps() const
{
    return apps;
}

const int AlgoFit2D::getBinCPUCapacity() const
{
    return bin_cpu_capacity;
}

const int AlgoFit2D::getBinMemCapacity() const
{
    return bin_mem_capacity;
}

const std::string& AlgoFit2D::getInstanceName() const
{
    return instance_name;
}

void AlgoFit2D::setSolution(BinList2D& bins)
{
    clearSolution();
    this->bins = bins;
    solved = true;
}

void AlgoFit2D::clearSolution()
{
    solved = false;
    for (Bin2D* bin : bins)
    {
        if (bin != nullptr)
        {
            delete bin;
        }
    }
    bins.clear();
    next_bin_index = 0;
}

void AlgoFit2D::createNewBin()
{
    bins.push_back(new Bin2D(next_bin_index, bin_cpu_capacity, bin_mem_capacity));
    next_bin_index += 1;
}

// Generic algorithm based on first fit
void AlgoFit2D::allocateBatch(AppList2D::iterator first_app, AppList2D::iterator end_batch)
{
    Bin2D* curr_bin = nullptr;
    bool allocated = false;

    sortApps(first_app, end_batch);
    auto curr_app_it = first_app;
    while(curr_app_it != end_batch)
    {
        Application2D * app = *curr_app_it;
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

                    // This is a quick safe guard to avoid infinite loops and running out of memory
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


// Solve the whole instance at once
// The hint is an estimate on the number of bins to allocate
int AlgoFit2D::solveInstance(int hint_nb_bins)
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
int AlgoFit2D::solvePerBatch(int batch_size, int hint_nb_bins)
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
    AppList2D::iterator curr_app_it;
    AppList2D::iterator stop_it = apps.begin();
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
Algo2DFF::Algo2DFF(const Instance2D &instance):
    AlgoFit2D(instance)
{ }

void Algo2DFF::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it) { }
void Algo2DFF::sortBins() { }

bool Algo2DFF::checkItemToBin(Application2D* app, Bin2D* bin) const
{
    return (bin->doesItemFit(app->getCPUSize(), app->getMemorySize())) and (bin->isAffinityCompliant(app));
}

void Algo2DFF::addItemToBin(Application2D* app, int replica_id, Bin2D* bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);
}

/************ First Fit Decreasing Degree Affinity *********/
Algo2DFFDDegree::Algo2DFFDDegree(const Instance2D &instance):
    Algo2DFF(instance)
{ }

void Algo2DFFDDegree::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_total_degree_decreasing);
}



/************ First Fit Decreasing Average Affinity *********/
Algo2DFFDAvg::Algo2DFFDAvg(const Instance2D &instance):
    Algo2DFF(instance)
{ }

void Algo2DFFDAvg::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avg_size_decreasing);
}


/************ FFD Average with Exponential Weights Affinity *********/
Algo2DFFDAvgExpo::Algo2DFFDAvgExpo(const Instance2D &instance):
    Algo2DFF(instance)
{ }

void Algo2DFFDAvgExpo::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avgexpo_size_decreasing);
}


/************ First Fit Decreasing Max Affinity *********/
Algo2DFFDMax::Algo2DFFDMax(const Instance2D &instance):
    Algo2DFF(instance)
{ }

void Algo2DFFDMax::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_max_size_decreasing);
}


/************ First Fit Decreasing Surrogate Affinity *********/
Algo2DFFDSurrogate::Algo2DFFDSurrogate(const Instance2D &instance):
    Algo2DFF(instance)
{ }

void Algo2DFFDSurrogate::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_surrogate_size_decreasing);
}


/************ First Fit Decreasing Extended Sum Affinity *********/
Algo2DFFDExtendedSum::Algo2DFFDExtendedSum(const Instance2D &instance):
    Algo2DFF(instance)
{ }

void Algo2DFFDExtendedSum::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_extsum_size_decreasing);
}




/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ Best Fit Decreasing Average Affinity *********/
Algo2DBFDAvg::Algo2DBFDAvg(const Instance2D &instance):
    AlgoFit2D(instance)
{ }

void Algo2DBFDAvg::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avg_size_decreasing);
}

void Algo2DBFDAvg::sortBins() {
    // The measure of the bins should have been updated before
    // Only need to sort the bins from current onward
    // Bins before curr_bin_index cannot accomodate replicas of the current application
    bubble_bin_up(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_increasing);
}

bool Algo2DBFDAvg::checkItemToBin(Application2D* app, Bin2D* bin) const
{
    return (bin->doesItemFit(app->getCPUSize(), app->getMemorySize())) and (bin->isAffinityCompliant(app));
}

void Algo2DBFDAvg::addItemToBin(Application2D* app, int replica_id, Bin2D* bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);

    updateBinMeasure(bin);
}

void Algo2DBFDAvg::updateBinMeasure(Bin2D *bin)
{
    // measure = normalised residual cpu + normalised residual memory
    float measure = (bin->getAvailableCPUCap() / bin->getMaxCPUCap()) + (bin->getAvailableMemCap() / bin->getMaxMemCap());
    bin->setMeasure(measure);
}


/************ Best Fit Decreasing Max Affinity *********/
Algo2DBFDMax::Algo2DBFDMax(const Instance2D &instance):
    Algo2DBFDAvg(instance)
{ }

void Algo2DBFDMax::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_max_size_decreasing);
}

void Algo2DBFDMax::updateBinMeasure(Bin2D *bin)
{
    // measure = max(normalised residual cpu, normalised residual memory)
    float measure = std::max((bin->getAvailableCPUCap() / bin->getMaxCPUCap()), (bin->getAvailableMemCap() / bin->getMaxMemCap()));
    bin->setMeasure(measure);
}


/************ Best Average with Exponential Weights Affinity *********/
Algo2DBFDAvgExpo::Algo2DBFDAvgExpo(const Instance2D &instance):
    Algo2DBFDAvg(instance),
    total_residual_cpu(0),
    total_residual_mem(0)
{ }

void Algo2DBFDAvgExpo::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avgexpo_size_decreasing);
}

void Algo2DBFDAvgExpo::sortBins() {
    // The measure of the bins should have been updated before
    // Need to sort all bins since all measures have changed
    stable_sort(bins.begin()+curr_bin_index, bins.end(), bin2D_comparator_measure_increasing);
}

void Algo2DBFDAvgExpo::createNewBin()
{
    //std::cout << "Opening a new bin of index " << next_bin_id << std::endl; //TODO remove
    bins.push_back(new Bin2D(next_bin_index, bin_cpu_capacity, bin_mem_capacity));
    next_bin_index += 1;

    total_residual_cpu += bin_cpu_capacity;
    total_residual_mem += bin_mem_capacity;
}

void Algo2DBFDAvgExpo::addItemToBin(Application2D *app, int replica_id, Bin2D *bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);

    total_residual_cpu -= app->getCPUSize();
    total_residual_mem -= app->getMemorySize();

    updateBinMeasure(bin);
}

void Algo2DBFDAvgExpo::updateBinMeasure(Bin2D *bin)
{
    // measure = exp(0.01* (sum residual capacity all bins)/(nb bin * bin capacity) * normalised residual cpu + same with memory
    float factor_cpu = (std::exp(0.01 * total_residual_cpu / (bin_cpu_capacity * bins.size()))) / bin_cpu_capacity;
    float factor_mem = (std::exp(0.01 * total_residual_mem / (bin_mem_capacity * bins.size()))) / bin_mem_capacity;

    for(auto it_bin = (bins.begin()+curr_bin_index); it_bin != bins.end(); ++it_bin)
    {
        float measure = factor_cpu * (*it_bin)->getAvailableCPUCap() + factor_mem * (*it_bin)->getAvailableMemCap();
        (*it_bin)->setMeasure(measure);
    }
}


/************ Best Fit Decreasing Surrogate Affinity *********/
Algo2DBFDSurrogate::Algo2DBFDSurrogate(const Instance2D &instance):
    Algo2DBFDAvgExpo(instance)
{ }

void Algo2DBFDSurrogate::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_surrogate_size_decreasing);
}

void Algo2DBFDSurrogate::sortBins() {
    // The measure of the bins should have been updated before
    // Need to sort all bins since all measures have changed
    stable_sort(bins.begin()+curr_bin_index, bins.end(), bin2D_comparator_measure_increasing);
}

void Algo2DBFDSurrogate::updateBinMeasure(Bin2D *bin)
{
    // measure = lambda norm residual cpu + (1-lambda) * norm residual mem
    float lambda = ((float)total_residual_cpu) / (total_residual_cpu + total_residual_mem);

    for(auto it_bin = (bins.begin()+curr_bin_index); it_bin != bins.end(); ++it_bin)
    {
        float measure = lambda * ((*it_bin)->getAvailableCPUCap() / bin_cpu_capacity) + (1-lambda) * ((*it_bin)->getAvailableMemCap() / bin_mem_capacity);
        (*it_bin)->setMeasure(measure);
    }
}


/************ Best Fit Decreasing Extended Sum Affinity *********/
Algo2DBFDExtendedSum::Algo2DBFDExtendedSum(const Instance2D &instance):
    Algo2DBFDAvgExpo(instance)
{ }

void Algo2DBFDExtendedSum::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_extsum_size_decreasing);
}

void Algo2DBFDExtendedSum::sortBins() {
    // The measure of the bins should have been updated before
    // Need to sort all bins since all measures have changed
    stable_sort(bins.begin()+curr_bin_index, bins.end(), bin2D_comparator_measure_increasing);
}

void Algo2DBFDExtendedSum::updateBinMeasure(Bin2D *bin)
{
    // measure = residual cpu / total residual cpu + residual mem / total residual mem
    // (no need to use normalised values here)
    for(auto it_bin = (bins.begin()+curr_bin_index); it_bin != bins.end(); ++it_bin)
    {
        float measure = ((float)(*it_bin)->getAvailableCPUCap()) / total_residual_cpu + ((float)(*it_bin)->getAvailableMemCap()) / total_residual_mem;
        (*it_bin)->setMeasure(measure);
    }
}


/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ Worst Fit Decreasing Avg Affinity *********/
Algo2DWFDAvg::Algo2DWFDAvg(const Instance2D &instance):
    Algo2DBFDAvg(instance)
{ }

void Algo2DWFDAvg::sortBins() {
    bubble_bin_up(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
}

/************ Worst Fit Decreasing Max Affinity *********/
Algo2DWFDMax::Algo2DWFDMax(const Instance2D &instance):
    Algo2DBFDMax(instance)
{ }

void Algo2DWFDMax::sortBins() {
    //bubble_bin_up(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
    stable_sort(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
}


/************ Worst Fit Decreasing AvgExpo Affinity *********/
Algo2DWFDAvgExpo::Algo2DWFDAvgExpo(const Instance2D &instance):
    Algo2DBFDAvgExpo(instance)
{ }

void Algo2DWFDAvgExpo::sortBins() {
    stable_sort(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
}


/************ Worst Fit Decreasing Surrogate Affinity *********/
Algo2DWFDSurrogate::Algo2DWFDSurrogate(const Instance2D &instance):
    Algo2DBFDSurrogate(instance)
{ }

void Algo2DWFDSurrogate::sortBins() {
    stable_sort(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
}


/************ Worst Fit Decreasing ExtendedSum Affinity *********/
Algo2DWFDExtendedSum::Algo2DWFDExtendedSum(const Instance2D &instance):
    Algo2DBFDExtendedSum(instance)
{ }

void Algo2DWFDExtendedSum::sortBins() {
    stable_sort(bins.begin() + curr_bin_index, bins.end(), bin2D_comparator_measure_decreasing);
}


/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ Medea Node Count Affinity *********/
Algo2DNodeCount::Algo2DNodeCount(const Instance2D &instance):
    AlgoFit2D(instance)
{ }

void Algo2DNodeCount::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it) { }
void Algo2DNodeCount::sortBins() { }

bool Algo2DNodeCount::checkItemToBin(Application2D* app, Bin2D* bin) const
{
    return (bin->doesItemFit(app->getCPUSize(), app->getMemorySize())) and (bin->isAffinityCompliant(app));
}

void Algo2DNodeCount::addItemToBin(Application2D* app, int replica_id, Bin2D* bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);
}


void Algo2DNodeCount::allocateBatch(AppList2D::iterator first_app, AppList2D::iterator end_batch)
{
    Bin2D* curr_bin = nullptr;
    bool allocated = false;

    //First sort the items in decreasing degree
    std::stable_sort(first_app, end_batch, application2D_comparator_total_degree_decreasing);

    // Brutal way to keep track of possible candidates, can probably be optimised
    // Stores for each app id the set of bin candidates (in which a replica can be packed)
    // The allocateBatch function can be called with a partial allocation of apps into bins
    // So initiate the list of bin candidates accordingly
    std::unordered_map<std::string, std::vector<int>> bin_candidates;
    for (Application2D* app : apps)
    {
        std::vector<int> v;
        for(Bin2D* bin : bins)
        {
            if(checkItemToBin(app, bin))
            {
                v.push_back(bin->getId());
            }
        }
        bin_candidates[app->getId()] = v;
    }

    auto current_app = first_app;
    std::string current_app_id;
    auto end_list = end_batch;
    while(current_app != end_list)
    {
        current_app_id = (*current_app)->getId();
        // Pack current app into bins
        std::vector<Bin2D*> bins_set; // The set of bins in which this item is packed
        auto next_app = current_app+1;

        auto bin_index_it = bin_candidates[current_app_id].begin();
        for (int j = 0; j < (*current_app)->getNbReplicas(); ++j)
        {
            // First try bin candidates
            allocated = false;
            while ( (!allocated) and bin_index_it != bin_candidates[current_app_id].end())
            {
                curr_bin = bins.at(*bin_index_it);
                if (checkItemToBin((*current_app), curr_bin))
                {
                    // This depends whether to update conflicts/affinities of the bin
                    addItemToBin((*current_app), j, curr_bin);
                    allocated = true;

                    // Update the bins set
                    if(std::find(bins_set.begin(), bins_set.end(), curr_bin) == bins_set.end())
                    {
                        bins_set.push_back(curr_bin);
                    }
                }
                else
                {
                    bin_index_it++;
                }
            }
            // Replica was packed or there are no more bin candidates

            if (!allocated)
            {
                // Create a new bin and pack the replica in it
                curr_bin = new Bin2D(next_bin_index, bin_cpu_capacity, bin_mem_capacity);
                bins.push_back(curr_bin);

                // This is a quick safe guard to avoid infinite loops and running out of memory
                if (bins.size() > total_replicas)
                {
                    std::cout << "There seem to be a problem with instance " << getInstanceName() << ", created " << bins.size() << " bins" << std::endl; //TODO remove
                    return;
                }

                // Add this bin to the candidates of all remaining apps
                for (auto it = next_app; it != end_list; ++it)
                {
                    bin_candidates[(*it)->getId()].push_back(next_bin_index);
                    (*it)->setMeasure(bin_candidates[(*it)->getId()].size());
                }
                bin_candidates[current_app_id].push_back(next_bin_index);

                next_bin_index += 1;

                // Update the iterator to point to the new bin
                bin_index_it = bin_candidates[current_app_id].end();
                bin_index_it--;

                // Ths bin is empty, pack the replica
                addItemToBin((*current_app), j, curr_bin);
                bins_set.push_back(curr_bin);
            }

        } // End for: All replicas of this item were packed
        (*current_app)->setFullyPacked(true);

        // Update the set of bin candidates of each adjacent item and their specific degree
        for (auto pair : (*current_app)->getAffinityInMap())
        {
            Application2D* app = getApp2D(apps, pair.first);
            if (!app->isFullyPacked()) // Otherwise don't need to update
            {
                for (Bin2D* bin : bins_set)
                {
                    std::vector<int>& bin_vect = bin_candidates[pair.first];
                    if (!checkItemToBin(app, bin))
                    {
                        // The adjacent item can no longer be packed, remove the bin from its candidates
                        auto it = std::find(bin_vect.begin(), bin_vect.end(), bin->getId());
                        if (it != bin_vect.end())
                        {
                            bin_vect.erase(it);
                            app->setMeasure(bin_vect.size());
                        }
                    }

                }
            }
        }
        for (auto pair : (*current_app)->getAffinityOutMap())
        {
            Application2D* app = getApp2D(apps, pair.first);
            if(!app->isFullyPacked())
            {
                for (Bin2D* bin : bins_set)
                {
                    std::vector<int>& bin_vect = bin_candidates[pair.first];
                    if (!checkItemToBin(app, bin))
                    {
                        // The adjacent item can no longer be packed, remove the bin from its candidates
                        auto it = std::find(bin_vect.begin(), bin_vect.end(), bin->getId());
                        if (it != bin_vect.end())
                        {
                            bin_vect.erase(it);
                            app->setMeasure(bin_vect.size());
                        }
                    }

                }
            }
        }
        // Then bubble up the item of smallest degree (smallest number of bin candidates)
        // And advance the item iterator
        bubble_apps2D_up(next_app, end_list, application2D_comparator_measure_increasing);

        current_app = next_app;
    }
}


/* ================================================ */
/* ================================================ */
/* ================================================ */
/********* Bin Centric FFD DotProduct ***************/
Algo2DBinFFDDotProduct::Algo2DBinFFDDotProduct(const Instance2D &instance):
    Algo2DFF(instance)
{ }

bool Algo2DBinFFDDotProduct::isBinFilled(Bin2D* bin)
{
    return ((bin->getAvailableCPUCap() == 0)
         or (bin->getAvailableMemCap() == 0));
}

void Algo2DBinFFDDotProduct::computeMeasures(AppList2D::iterator start_list, AppList2D::iterator end_list, Bin2D *bin)
{
    for(auto it = start_list; it != end_list; ++it)
    {
        Application2D * app = *it;
        // Use normalized values of app size and bin residual capacity
        float measure = (app->getNormalizedCPU() * bin->getAvailableCPUCap()) / bin->getMaxCPUCap();
        measure += (app->getNormalizedMemory() * bin->getAvailableMemCap()) / bin->getMaxMemCap();
        app->setMeasure(measure);
    }
}

Bin2D* Algo2DBinFFDDotProduct::createNewBinRet()
{
    Bin2D* bin = new Bin2D(next_bin_index, bin_cpu_capacity, bin_mem_capacity);
    next_bin_index += 1;
    bins.push_back(bin);
    return bin;
}

void Algo2DBinFFDDotProduct::allocateBatch(AppList2D::iterator first_app, AppList2D::iterator end_batch)
{
    std::unordered_map<std::string, int> next_id_replicas;
    int nb_apps = end_batch - first_app;
    next_id_replicas.reserve(nb_apps);// Stores the id of the next replica to pack for each application

    for(auto it = first_app; it != end_batch; ++it)
    {
        next_id_replicas[(*it)->getId()] = 0;
    }

    Bin2D* curr_bin = nullptr;

    // While there are still applications to pack
    auto next_treated_app_it = first_app;
    auto end_list = end_batch; // TODO probably don't need it, simply replace by end_batch
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
            bubble_apps2D_up(current_app_it, end_list, application2D_comparator_measure_decreasing);

            // Retrieve the next application to pack
            Application2D* app = *current_app_it;

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
Algo2DBinFFDDotDivision::Algo2DBinFFDDotDivision(const Instance2D &instance):
    Algo2DBinFFDDotProduct(instance)
{ }

void Algo2DBinFFDDotDivision::computeMeasures(AppList2D::iterator start_list, AppList2D::iterator end_list, Bin2D *bin)
{
    for(auto it = start_list; it != end_list; ++it)
    {
        Application2D * app = *it;
        // Use normalized values of app size and bin residual capacity
        float measure = (app->getNormalizedCPU() * bin->getMaxCPUCap()) / bin->getAvailableCPUCap();
        measure += (app->getNormalizedMemory() * bin->getMaxMemCap()) / bin->getAvailableMemCap();
        app->setMeasure(measure);
    }
}


/********* Bin Centric FFD L2Norm ***************/
Algo2DBinFFDL2Norm::Algo2DBinFFDL2Norm(const Instance2D &instance):
    Algo2DBinFFDDotProduct(instance)
{ }

void Algo2DBinFFDL2Norm::computeMeasures(AppList2D::iterator start_list, AppList2D::iterator end_list, Bin2D *bin)
{
    for(auto it = start_list; it != end_list; ++it)
    {
        Application2D * app = *it;
        // Use normalized values of app size and bin residual capacity
        float a = (bin->getAvailableCPUCap() / bin->getMaxCPUCap()) - app->getNormalizedCPU();
        float b = (bin->getAvailableMemCap() / bin->getMaxMemCap()) - app->getNormalizedMemory();
        float measure = - a*a + b*b;
        app->setMeasure(measure);
    }
}



/********* Bin Centric FFD Fitness ***************/
Algo2DBinFFDFitness::Algo2DBinFFDFitness(const Instance2D &instance):
    Algo2DBinFFDDotProduct(instance),
    total_residual_cpu(0),
    total_residual_mem(0)
{ }

Bin2D* Algo2DBinFFDFitness::createNewBinRet()
{
    Bin2D* bin = new Bin2D(next_bin_index, bin_cpu_capacity, bin_mem_capacity);
    next_bin_index += 1;
    bins.push_back(bin);

    total_residual_cpu += bin_cpu_capacity;
    total_residual_mem += bin_mem_capacity;

    return bin;
}

void Algo2DBinFFDFitness::addItemToBin(Application2D *app, int replica_id, Bin2D *bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);

    total_residual_cpu -= app->getCPUSize();
    total_residual_mem -= app->getMemorySize();
}


void Algo2DBinFFDFitness::computeMeasures(AppList2D::iterator start_list, AppList2D::iterator end_list, Bin2D *bin)
{
    for(auto it = start_list; it != end_list; ++it)
    {
        Application2D * app = *it;
        float a = (app->getNormalizedCPU() * bin->getAvailableCPUCap()) / (norm_sum_cpu * total_residual_cpu);
        float b = (app->getNormalizedMemory() * bin->getAvailableMemCap()) / (norm_sum_mem * total_residual_mem);

        app->setMeasure(a+b);
    }
}



/* ================================================ */
/* ================================================ */
/* ================================================ */
/*********** Spread replicas Worst Fit Avg **********/
Algo2DSpreadWFDAvg::Algo2DSpreadWFDAvg(const Instance2D &instance):
    AlgoFit2D(instance)
{ }

int Algo2DSpreadWFDAvg::solveInstanceSpread(int LB_bins, int UB_bins)
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
            i++;
        }
    }

    // Store the current solution
    BinList2D best_bins = getBinsCopy();
    int best_sol = UB_bins;
    int low_bound = LB_bins;
    int target_bins;

    // Then iteratively try to improve on the solution
    int nb_iter = 0;
    while(low_bound < best_sol)
    {
        //target_bins = (low_bound + 3*best_sol)/4;
        //target_bins = (3*low_bound + best_sol)/4;
        target_bins = (low_bound + best_sol)/2;
        //std::cout << "Trying with " << target_bins << " bins (LB: " << low_bound  << " UB: " << best_sol<< ")" << std::endl;

        if (trySolve(target_bins))
        {

            // Update the best solution
            best_sol = target_bins;
            for (Bin2D* bin : best_bins)
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
        nb_iter+=1;
    }
    setSolution(best_bins);
    //std::cout << "Found best solution in " << nb_iter << " iterations" << std::endl;
    return best_sol;
}

bool Algo2DSpreadWFDAvg::trySolve(int nb_bins)
{
    clearSolution();
    createBins(nb_bins);

    // For each app in the list, try to put all replicas in separate bins
    Bin2D* curr_bin = nullptr;
    sortApps(apps.begin(), apps.end());
    auto current_app_it = apps.begin();
    while(current_app_it != apps.end())
    {
        Application2D * app = *current_app_it;
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

void Algo2DSpreadWFDAvg::createBins(int nb_bins)
{
    bins.reserve(nb_bins);
    for (int i = 0; i < nb_bins; ++i)
    {
        Bin2D* bin = new Bin2D(i, bin_cpu_capacity, bin_mem_capacity);
        updateBinMeasure(bin);
        bins.push_back(bin);
    }
}


void Algo2DSpreadWFDAvg::updateBinMeasure(Bin2D* bin)
{
    float measure = (bin->getAvailableCPUCap() / bin->getMaxCPUCap()) + (bin->getAvailableMemCap() / bin->getMaxMemCap());
    bin->setMeasure(measure);
}

void Algo2DSpreadWFDAvg::updateBinMeasures(){ }


void Algo2DSpreadWFDAvg::allocateBatch(AppList2D::iterator first_app, AppList2D::iterator end_batch)
{
    std::cout << "For Spread algorithm please call 'solveInstanceSpread' instead" << std::endl;
    return;
}

void Algo2DSpreadWFDAvg::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avg_size_decreasing);
}

void Algo2DSpreadWFDAvg::sortBins()
{
    stable_sort(bins.begin(), bins.end(), bin2D_comparator_measure_decreasing);
}

bool Algo2DSpreadWFDAvg::checkItemToBin(Application2D* app, Bin2D* bin) const
{
    return (bin->doesItemFit(app->getCPUSize(), app->getMemorySize())) and (bin->isAffinityCompliant(app));
}

void Algo2DSpreadWFDAvg::addItemToBin(Application2D* app, int replica_id, Bin2D* bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);
}



/*********** Spread replicas Worst Fit Max **********/
Algo2DSpreadWFDMax::Algo2DSpreadWFDMax(const Instance2D &instance):
    Algo2DSpreadWFDAvg(instance)
{ }

void Algo2DSpreadWFDMax::updateBinMeasure(Bin2D* bin)
{
    float measure = std::max(bin->getAvailableCPUCap() / bin->getMaxCPUCap(), bin->getAvailableMemCap() / bin->getMaxMemCap());
    bin->setMeasure(measure);
}

void Algo2DSpreadWFDMax::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_max_size_decreasing);
}



/*********** Spread replicas Worst Fit AvgExpo **********/
Algo2DSpreadWFDAvgExpo::Algo2DSpreadWFDAvgExpo(const Instance2D &instance):
    Algo2DSpreadWFDAvg(instance),
    total_residual_cpu(0),
    total_residual_mem(0)
{ }

void Algo2DSpreadWFDAvgExpo::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_avgexpo_size_decreasing);
}

void Algo2DSpreadWFDAvgExpo::createBins(int nb_bins)
{
    bins.reserve(nb_bins);
    for (int i = 0; i < nb_bins; ++i)
    {
        Bin2D* bin = new Bin2D(i, bin_cpu_capacity, bin_mem_capacity);
        bins.push_back(bin);
    }

    total_residual_cpu = nb_bins * bin_cpu_capacity;
    total_residual_mem = nb_bins * bin_mem_capacity;
}

void Algo2DSpreadWFDAvgExpo::addItemToBin(Application2D* app, int replica_id, Bin2D* bin)
{
    bin->addNewConflict(app);
    bin->addItem(app, replica_id);

    total_residual_cpu -= app->getCPUSize();
    total_residual_mem -= app->getMemorySize();
}

void Algo2DSpreadWFDAvgExpo::updateBinMeasure(Bin2D* bin) { }

void Algo2DSpreadWFDAvgExpo::updateBinMeasures()
{
    // measure = exp(0.01* (sum residual capacity all bins)/(nb bin * bin capacity) * normalised residual cpu + same with memory
    float factor_cpu = (std::exp(0.01 * total_residual_cpu / (bin_cpu_capacity * bins.size()))) / bin_cpu_capacity;
    float factor_mem = (std::exp(0.01 * total_residual_mem / (bin_mem_capacity * bins.size()))) / bin_mem_capacity;

    for(auto it_bin = bins.begin(); it_bin != bins.end(); ++it_bin)
    {
        float measure = factor_cpu * (*it_bin)->getAvailableCPUCap() + factor_mem * (*it_bin)->getAvailableMemCap();
        (*it_bin)->setMeasure(measure);
    }
}




/*********** Spread replicas Worst Fit Surrogate **********/
Algo2DSpreadWFDSurrogate::Algo2DSpreadWFDSurrogate(const Instance2D &instance):
    Algo2DSpreadWFDAvgExpo(instance)
{ }

void Algo2DSpreadWFDSurrogate::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_max_size_decreasing);
}

void Algo2DSpreadWFDSurrogate::updateBinMeasure(Bin2D* bin) { }

void Algo2DSpreadWFDSurrogate::updateBinMeasures()
{
    // measure = lambda norm residual cpu + (1-lambda) * norm residual mem
    float lambda = ((float)total_residual_cpu) / (total_residual_cpu + total_residual_mem);

    for(auto it_bin = bins.begin(); it_bin != bins.end(); ++it_bin)
    {
        float measure = lambda * ((*it_bin)->getAvailableCPUCap() / bin_cpu_capacity) + (1-lambda) * ((*it_bin)->getAvailableMemCap() / bin_mem_capacity);
        (*it_bin)->setMeasure(measure);
    }
}



/*********** Spread replicas Worst Fit Extended Sum **********/
Algo2DSpreadWFDExtendedSum::Algo2DSpreadWFDExtendedSum(const Instance2D &instance):
    Algo2DSpreadWFDAvgExpo(instance)
{ }

void Algo2DSpreadWFDExtendedSum::sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it)
{
    stable_sort(first_app, end_it, application2D_comparator_max_size_decreasing);
}

void Algo2DSpreadWFDExtendedSum::updateBinMeasure(Bin2D* bin) { }

void Algo2DSpreadWFDExtendedSum::updateBinMeasures()
{
    // measure = residual cpu / total residual cpu + residual mem / total residual mem
    // (no need to use normalised values here)
    for(auto it_bin = bins.begin(); it_bin != bins.end(); ++it_bin)
    {
        float measure = ((float)(*it_bin)->getAvailableCPUCap()) / total_residual_cpu + ((float)(*it_bin)->getAvailableMemCap()) / total_residual_mem;
        (*it_bin)->setMeasure(measure);
    }
}

#include "application.hpp"

#include <cmath>
#include <unordered_set>

Application2D::Application2D(std::string& app_id, int internal_id,
              int nb_replicas, int nb_cpus, int nb_memory,
              int affinity_degree, AffinityMap& affinities_out):
    id(app_id),
    internal_id(internal_id),
    nb_replicas(nb_replicas),
    nb_cpus(nb_cpus),
    nb_memory(nb_memory),
    affinity_out_degree(affinity_degree),
    affinity_out_map(affinities_out),
    measure(0.0)
{ }

const std::string& Application2D::getId() const
{
    return id;
}

const int Application2D::getInternalId() const
{
    return internal_id;
}

const int Application2D::getCPUSize() const
{
    return nb_cpus;
}

const int Application2D::getMemorySize() const
{
    return nb_memory;
}

const float Application2D::getNormalizedMemory() const
{
    return norm_memory;
}

const float Application2D::getNormalizedCPU() const
{
    return norm_cpus;
}

const int Application2D::getOutDegree() const
{
    return affinity_out_degree;
}

const int Application2D::getTotalDegree() const
{
    return affinity_total_degree;
}

const int Application2D::getNbReplicas() const
{
    return nb_replicas;
}

const AffinityMap& Application2D::getAffinityOutMap() const
{
    return affinity_out_map;
}

const AffinityMap& Application2D::getAffinityInMap() const
{
    return affinity_in_map;
}

void Application2D::removeAppsAffinity(std::vector<std::string>& to_remove)
{
    for (std::string& app_str : to_remove)
    {
        affinity_in_map.erase(app_str);
        affinity_out_map.erase(app_str);
    }
    affinity_out_degree = affinity_out_map.size();
}

void Application2D::setAffinityInMap(AffinityMap& affinities_in)
{
    affinity_in_map = affinities_in;

    std::unordered_set<std::string> neighbours;
    for (auto pair : affinity_out_map)
    {
        neighbours.insert(pair.first);
    }
    for(auto pair : affinity_in_map)
    {
        neighbours.insert(pair.first);
    }
    affinity_total_degree = neighbours.size();
}


std::string Application2D::toString(bool full) const
{
    std::string s(id);
    s+= ": " + std::to_string(nb_replicas) + "\treplicas, " + std::to_string(nb_cpus) + " cores, " + std::to_string(nb_memory) + " memory and degree " + std::to_string(affinity_out_degree);
    if (full)
    {
        s+= ":\n\t";
        for (auto pair : affinity_out_map)
        {
            s+= "(" + pair.first + ", " + std::to_string(pair.second) + "), ";
        }
    }
    return s;
}

void Application2D::setParams(float sum_cpu, float sum_mem, int total_replicas,
                              int bin_cpu_cap, int bin_mem_cap)
{
    norm_cpus = nb_cpus / bin_cpu_cap;      // Normalized size
    norm_memory = nb_memory / bin_mem_cap;  // Normalized size
    float lambda = sum_cpu / (sum_cpu + sum_mem); // Lamba (for surrogate)
    float weight_cpu = sum_cpu / (total_replicas * bin_cpu_cap); // Avg normalized cpu size
    float weight_mem = sum_mem / (total_replicas * bin_mem_cap); // Avg normalized mem size
    surrogate_size = lambda * norm_cpus + (1-lambda) * norm_memory;
    ext_sum_size = ((nb_replicas*nb_cpus)/sum_cpu) + ((nb_replicas*nb_memory)/sum_mem); // Don't use normalised values because sum_cpu and sum_mem are not normalised
    avg_size = norm_cpus + norm_memory; // No need to divide by 2
    max_size = std::max(norm_cpus, norm_memory);

    avg_expo_size = std::exp(0.01 * weight_cpu) * norm_cpus + std::exp(0.01 * weight_mem) * norm_memory;
}


const float Application2D::getSurrogate() const
{
    return surrogate_size;
}

const float Application2D::getExtSum() const
{
    return ext_sum_size;
}

const float Application2D::getAvgSize() const
{
    return avg_size;
}

const float Application2D::getMaxSize() const
{
    return max_size;
}

const float Application2D::getAvgExpoSize() const
{
    return avg_expo_size;
}

void Application2D::setMeasure(float measure)
{
    this->measure = measure;
}

const float Application2D::getMeasure() const
{
    return measure;
}

void Application2D::setFullyPacked(bool val)
{
    fully_packed = val;
}

const bool Application2D::isFullyPacked() const
{
    return fully_packed;
}



Application2D* getApp2D(const AppList2D& list, const std::string& app_id)
{
    auto it = list.begin();
    while (it != list.end())
    {
        if ((*it)->getId() == app_id)
        {
            return *it;
        }
        ++it;
    }
    return nullptr;
}


bool application2D_comparator_total_degree_decreasing(Application2D* appa, Application2D* appb)
{
    return (appa->getTotalDegree() > appb->getTotalDegree());
}

bool application2D_comparator_max_size_decreasing(Application2D* appa, Application2D* appb)
{
    return (appa->getMaxSize() > appb->getMaxSize());
}

bool application2D_comparator_avg_size_decreasing(Application2D* appa, Application2D* appb)
{
    return (appa->getAvgSize() > appb->getAvgSize());
}

bool application2D_comparator_surrogate_size_decreasing(Application2D* appa, Application2D* appb)
{
    return (appa->getSurrogate() > appb->getSurrogate());
}

bool application2D_comparator_extsum_size_decreasing(Application2D* appa, Application2D* appb)
{
    return (appa->getExtSum() > appb->getExtSum());
}

bool application2D_comparator_avgexpo_size_decreasing(Application2D* appa, Application2D* appb)
{
    return (appa->getAvgExpoSize() > appb->getAvgExpoSize());
}

bool application2D_comparator_measure_increasing(Application2D* appa, Application2D* appb)
{
    return(appa->getMeasure() < appb->getMeasure());
}

bool application2D_comparator_measure_decreasing(Application2D* appa, Application2D* appb)
{
    return(appa->getMeasure() > appb->getMeasure());
}


// Perform one round of bubble downwards
void bubble_apps2D_down(AppList2D::iterator first, AppList2D::iterator last, bool comp (Application2D*, Application2D*))
{
    if (first == last) // The vector is empty...
        return;

    auto next = first;
    next++;
    if (next == last) // there is only one element in the vector
        return;

    auto current = first;
    while(next != last)
    {
        if (comp(*next, *current))
        {
            std::iter_swap(current, next);
        }
        ++current;
        ++next;
    }
}

// Perform one round of bubble upwards
void bubble_apps2D_up(AppList2D::iterator first, AppList2D::iterator last, bool comp (Application2D*, Application2D*))
{

    if (first == last)
        return; // Maybe first is also at the end
    --last; // last MUST point to the end of the vector
    // In case only one element
    if (first == last)
        return;

    auto current = last;
    auto previous = last-1;
    while(first != previous)
    {
        if (comp(*current, *previous))
        {
            std::iter_swap(current, previous);
        }
        --current;
        --previous;
    }
    // One last time at the head of the vector
    if (comp(*current, *previous))
    {
        std::iter_swap(current, previous);
    }
}





ApplicationTS::ApplicationTS(std::string& app_id, int internal_id,
                             int nb_replicas, size_t size_TS,
                             ResourceTS& cpu_usage, ResourceTS& mem_usage,
                             float peak_cpu, float peak_mem,
                             int affinity_degree, AffinityMap& affinities):
    Application2D(app_id, internal_id, nb_replicas, 0, 0, affinity_degree, affinities),
    TS_size(size_TS),
    cpu_usage(cpu_usage),
    mem_usage(mem_usage),
    peak_cpu(peak_cpu),
    peak_mem(peak_mem),
    norm_cpus_TS(size_TS, 0.0),
    norm_memory_TS(size_TS, 0.0)
{ }


const ResourceTS& ApplicationTS::getCpuUsage() const
{
    return cpu_usage;
}

const ResourceTS& ApplicationTS::getMemUsage() const
{
    return mem_usage;
}

const ResourceTS& ApplicationTS::getNormCpuUsage() const
{
    return norm_cpus_TS;
}

const ResourceTS& ApplicationTS::getNormMemUsage() const
{
    return norm_memory_TS;
}

void ApplicationTS::setParams(ResourceTS& sum_cpu, ResourceTS& sum_mem,
                              float total_sum_cpu_mem,
                              int total_replicas,
                              int bin_cpu_cap, int bin_mem_cap)
{
    ResourceTS lambda_cpu(TS_size, 0.0);
    ResourceTS lambda_mem(TS_size, 0.0);
    ResourceTS weight_cpu(TS_size, 0.0);
    ResourceTS weight_mem(TS_size, 0.0);

    surrogate_size = 0.0;
    ext_sum_size = 0.0;
    avg_size = 0.0;
    avg_expo_size = 0.0;

    max_size = std::max((peak_cpu / bin_cpu_cap), (peak_mem / bin_mem_cap));

    // For each dimension in the time series
    for(int i = 0; i < TS_size; ++i)
    {
        norm_cpus_TS[i] = cpu_usage[i] / bin_cpu_cap;
        norm_memory_TS[i] = mem_usage[i] / bin_mem_cap;
        lambda_cpu[i] = (sum_cpu[i] / total_sum_cpu_mem);
        lambda_mem[i] = (sum_mem[i] / total_sum_cpu_mem);
        weight_cpu[i] = (sum_cpu[i] / (total_replicas * bin_cpu_cap));
        weight_mem[i] = (sum_mem[i] / (total_replicas * bin_mem_cap));

        surrogate_size += lambda_cpu[i] * norm_cpus_TS[i] + lambda_mem[i] * norm_memory_TS[i];
        ext_sum_size += ((nb_replicas*cpu_usage[i])/sum_cpu[i]) + ((nb_replicas*mem_usage[i])/sum_mem[i]);
        avg_size += norm_cpus_TS[i] + norm_memory_TS[i];
        avg_expo_size += std::exp(0.01 * weight_cpu[i]) * norm_cpus_TS[i] + std::exp(0.01 * weight_mem[i]) * norm_memory_TS[i];
    }
}

// Perform one round of bubble downwards
void bubble_appsTS_down(AppListTS::iterator first, AppListTS::iterator last, bool comp (Application2D*, Application2D*))
{
    if (first == last) // The vector is empty...
        return;

    auto next = first;
    next++;
    if (next == last) // there is only one element in the vector
        return;

    auto current = first;
    while(next != last)
    {
        if (comp(*next, *current))
        {
            std::iter_swap(current, next);
        }
        ++current;
        ++next;
    }
}

// Perform one round of bubble upwards
void bubble_appsTS_up(AppListTS::iterator first, AppListTS::iterator last, bool comp (Application2D*, Application2D*))
{

    if (first == last)
        return; // Maybe first is also at the end
    --last; // last MUST point to the end of the vector
    // In case only one element
    if (first == last)
        return;

    auto current = last;
    auto previous = last-1;
    while(first != previous)
    {
        if (comp(*current, *previous))
        {
            std::iter_swap(current, previous);
        }
        --current;
        --previous;
    }
    // One last time at the head of the vector
    if (comp(*current, *previous))
    {
        std::iter_swap(current, previous);
    }
}


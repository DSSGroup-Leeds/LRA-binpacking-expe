#include "bins.hpp"

#include <iostream>
#include <sstream>
#include <algorithm>

using namespace std;

Bin2D::Bin2D(int id, int max_cpu_capacity, int max_mem_capacity):
    id(id),
    max_cpu_capacity(max_cpu_capacity),
    max_mem_capacity(max_mem_capacity),
    available_cpu_capacity(max_cpu_capacity),
    available_mem_capacity(max_mem_capacity),
    measure(0.0)
{ }

const int Bin2D::getId() const
{
    return id;
}

const int Bin2D::getMaxCPUCap() const
{
    return max_cpu_capacity;
}

const int Bin2D::getAvailableCPUCap() const
{
    return available_cpu_capacity;
}

const int Bin2D::getMaxMemCap() const
{
    return max_mem_capacity;
}

const int Bin2D::getAvailableMemCap() const
{
    return available_mem_capacity;
}

const AllocMap& Bin2D::getAllocMap() const
{
    return alloc_map;
}

const ConflictMap& Bin2D::getConflictMap() const
{
    return conflict_map;
}


void Bin2D::addItem(Application2D* app, int replica_id)
{
    // We do not check anything and don't return a boolean
    // That's the job of the algo to not make stupid decisions.
    if (doesItemFit(app->getCPUSize(), app->getMemorySize()))
    {
        auto it = alloc_map.find(app->getId());
        if (it == alloc_map.end())
        {
            std::vector<int> v(1, replica_id);
            alloc_map.insert(it, {app->getId(), v});
        }
        else
        {
            it->second.push_back(replica_id);
        }
        available_cpu_capacity -= app->getCPUSize();

        available_mem_capacity -= app->getMemorySize();
    }
}

bool Bin2D::doesItemFit(int size_cpu, int size_mem) const
{
    return ((size_cpu <= available_cpu_capacity) and (size_mem <= available_mem_capacity));
}

void Bin2D::printAlloc() const
{
    stringstream ss;
    ss << "Bin_" << id << ": ";

    std::vector<std::string> keys;
    keys.reserve(alloc_map.size());
    for (auto app_it : alloc_map)
    {
        keys.push_back(app_it.first);
    }

    auto compare = [](const std::string& a, const std::string& b) {
        return std::stoi(a) < std::stoi(b);
    };

    sort(keys.begin(), keys.end(), compare);
    for (std::string& key: keys)
    {
        for (auto e : alloc_map.at(key))
        {
            ss << key << "_" << e << ",";
        }
    }
    cout << ss.str() << endl;
}


bool Bin2D::isAffinityCompliant(Application2D *app) const
{
    auto it = conflict_map.find(app->getId());
    if (it != conflict_map.end())
    {
        // The candidate app is in conflict with apps in the bin
        if (it->second < 1)
        {
            // 0 replicas of the candidate app are tolerated
            return false;
        }

        auto it2 = alloc_map.find(app->getId());
        if (it2 != alloc_map.end())
        {
            // There are already replicas of the candidate app
            // Check if we can put one more
            if (it->second < (it2->second.size() + 1))
            {
                return false;
            }
        }
    }
    for (auto pair : app->getAffinityOutMap())
    {
        // For each app_b in conflict with the candidate app
        // check if there are more than the tolerated replicas
        auto it3 = alloc_map.find(pair.first);
        if (it3 != alloc_map.end())
        {
            if (it3->second.size() > pair.second)
            {
                return false;
            }
        }
    }
    return true;
}


void Bin2D::addNewConflict(Application2D *app)
{
    // Only add conflicts if the app is new to the bin (i.e., there was no replica of the app yet in the bin)
    if (alloc_map.find(app->getId()) != alloc_map.end())
    {
        return;
    }

    for (auto pair : app->getAffinityOutMap())
    {
        auto it = conflict_map.find(pair.first);
        if (it != conflict_map.end())
        {
            it->second = min(pair.second, it->second);
        }
        else
        {
            conflict_map[pair.first] = pair.second;
        }
    }
}


void Bin2D::setMeasure(float measure)
{
    this->measure = measure;
}

const float Bin2D::getMeasure() const
{
    return measure;
}



bool bin2D_comparator_measure_increasing(Bin2D* bina, Bin2D* binb)
{
    return(bina->getMeasure() < binb->getMeasure());
}

// Perform one round of bubble upwards
void bubble_bin_up(BinList2D::iterator first, BinList2D::iterator last, bool comp(Bin2D*, Bin2D*))
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




BinTS::BinTS(int id, int max_cpu_capacity, int max_mem_capacity, size_t size_TS):
    Bin2D(id, max_cpu_capacity, max_mem_capacity),
    available_cpu_capacity(size_TS, max_cpu_capacity),
    available_mem_capacity(size_TS, max_mem_capacity),
    size_TS(size_TS),
    total_residual_cpu(0.0),
    total_residual_mem(0.0)
{ }


void BinTS::addItem(ApplicationTS* app, int replica_id)
{
    // We do not check anything and don't return a boolean
    // That's the job of the algo to not make stupid decisions.
    if (doesItemFit(app))
    {
        auto it = alloc_map.find(app->getId());
        if (it == alloc_map.end())
        {
            std::vector<int> v(1, replica_id);
            alloc_map.insert(it, {app->getId(), v});
        }
        else
        {
            it->second.push_back(replica_id);
        }

        // Update usage vectors
        const ResourceTS& app_cpu = app->getCpuUsage();
        const ResourceTS& app_mem = app->getMemUsage();

        for (size_t i = 0; i < size_TS; i++)
        {
            available_cpu_capacity[i] -= app_cpu[i];
            available_mem_capacity[i] -= app_mem[i];

            total_residual_cpu -= app_cpu[i];
            total_residual_mem -= app_mem[i];
        }
    }
}


bool BinTS::doesItemFit(ApplicationTS* app) const
{
    const ResourceTS& app_cpu = app->getCpuUsage();
    const ResourceTS& app_mem = app->getMemUsage();

    for (size_t i = 0; i < size_TS; i++)
    {
        if ((app_cpu[i] > available_cpu_capacity[i]) or
            (app_mem[i] > available_mem_capacity[i]) )
        {
            return false;
        }
    }
    return true;
}

const ResourceTS& BinTS::getAvailableCPUCaps() const
{
    return available_cpu_capacity;
}

const ResourceTS& BinTS::getAvailableMemCaps() const
{
    return available_mem_capacity;
}

const float BinTS::getTotalResidualCPU() const
{
    return total_residual_cpu;
}

const float BinTS::getTotalResidualMem() const
{
    return total_residual_mem;
}

// Perform one round of bubble upwards
void bubble_bin_up(BinListTS::iterator first, BinListTS::iterator last, bool comp(Bin2D*, Bin2D*))
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


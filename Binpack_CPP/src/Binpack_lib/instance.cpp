#include "instance.hpp"

#define CSV_IO_NO_THREAD // Disable multithreading in csv.h
#include "csv.h" // From https://github.com/ben-strasser/fast-cpp-csv-parser

#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

using namespace io; // From csv.h

void my_trim(std::string& s)
{
    s.erase(std::remove(s.begin(), s.end(), '('), s.end());
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    s.erase(std::remove(s.begin(), s.end(), ')'), s.end());
}

AffinityMap constructAffinitiyMap(std::string& aff_str)
{
    AffinityMap aff_map;
    std::string app_b, strk;
    std::istringstream splitStream(aff_str);
    while(std::getline(splitStream, app_b, ','))
    {
        my_trim(app_b);
        std::getline(splitStream, strk, ',');
        my_trim(strk);
        aff_map[app_b] = stoi(strk);
    }
    return aff_map;
}



Instance2D::Instance2D(std::string id, int bin_cpu_capacity, int bin_memory_capacity,
                       std::string& filename, int max_size)
{
    this->id = id;
    this->bin_cpu_capacity = bin_cpu_capacity;
    this->bin_mem_capacity = bin_memory_capacity;
    sum_cpu = 0;
    sum_mem = 0;
    total_replicas = 0;

    // Reading the CSV file
    CSVReader<6, trim_chars<'[', ']'>, no_quote_escape<'\t'>> reader(filename);
    reader.read_header(ignore_extra_column, "app_id", "nb_instances", "core", "memory", "inter_degree", "inter_aff");
    std::string app_id("");
    std::string aff_str("");
    int nb_rep, nb_cpus, nb_memory, degree;

    // Applications that do not fit in the bins must be removed
    // from the list and from the affinity maps
    std::vector<std::string> to_remove;

    // Mapps for each app id its affinity_in_map
    std::unordered_map<std::string, AffinityMap> maps_in;

    int internal_id = 0;
    // For each row create one Application
    while(reader.read_row(app_id, nb_rep, nb_cpus, nb_memory, degree, aff_str))
    {
        // Make sure the replicas can be allocated to bins
        if ( (nb_cpus <= bin_cpu_capacity) and (nb_memory <= bin_memory_capacity) and (internal_id < max_size))
        {
            // Retrieve the map of affinities from the affinity string
            AffinityMap aff_map_out = constructAffinitiyMap(aff_str);
            
            // Update the in map of other items
            for(auto pair : aff_map_out)
            {
                auto it = maps_in.find(pair.first);
                if (it != maps_in.end())
                {
                    maps_in[pair.first][app_id] = pair.second;
                }
                else
                {
                    maps_in[pair.first] = AffinityMap();
                    maps_in[pair.first][app_id] = pair.second;
                }
            }

            app_list.push_back(new Application2D(app_id, internal_id, nb_rep,
                                                 nb_cpus, nb_memory,
                                                 degree, aff_map_out));
            internal_id++;

            sum_cpu += nb_cpus * nb_rep;
            sum_mem += nb_memory * nb_rep;
            total_replicas += nb_rep;
        }
        else
        {
            // Else a replica does not fit in a bin, drop the application
            to_remove.push_back(app_id);
        }
    }

    // Remove the filtered out apps from the affinity map
    // of remaining apps
    for (Application2D* app : app_list)
    {
        app->setAffinityInMap(maps_in[app->getId()]);
        app->removeAppsAffinity(to_remove);
        app->setParams(sum_cpu, sum_mem, total_replicas, bin_cpu_capacity, bin_memory_capacity);
    }
}


Instance2D::~Instance2D()
{
    for (Application2D* app : app_list)
    {
        if (app != nullptr)
        {
            delete app;
        }
    }
}

const std::string& Instance2D::getId() const
{
    return id;
}

const int Instance2D::getBinCPUCapacity() const
{
    return bin_cpu_capacity;
}

const int Instance2D::getBinMemCapacity() const
{
    return bin_mem_capacity;
}

const AppList2D& Instance2D::getApps() const
{
    return app_list;
}

const int Instance2D::getSumCPU() const
{
    return sum_cpu;
}

const int Instance2D::getSumMem() const
{
    return sum_mem;
}

const int Instance2D::getTotalReplicas() const
{
    return total_replicas;
}




InstanceTS::InstanceTS(std::string id, int bin_cpu_capacity,
                       int bin_mem_capacity,
                       std::string& filename,
                       size_t size_series):
    id(id),
    bin_cpu_capacity(bin_cpu_capacity),
    bin_mem_capacity(bin_mem_capacity),
    TS_size(size_series),
    total_replicas(0),
    sum_cpu_TS(size_series, 0.0),
    sum_mem_TS(size_series, 0.0)
{
    // Reading the CSV file
    CSVReader<6, trim_chars<'[', ']'>, no_quote_escape<'\t'>> reader(filename);
    reader.read_header(ignore_extra_column, "app_id", "nb_instances", "core", "memory", "inter_degree", "inter_aff");
    std::string app_id("");
    std::string aff_str("");
    std::string cpu_usage_str;
    std::string mem_usage_str;
    int nb_rep, degree;

    // Applications that do not fit in the bins must be removed
    // from the list and from the affinity maps
    std::vector<std::string> to_remove;

    // Mapps for each app id its affinity_in_map
    std::unordered_map<std::string, AffinityMap> maps_in;

    float total_sum_cpu_mem = 0.0;

    int internal_id = 0;
    // For each row create one Application
    while(reader.read_row(app_id, nb_rep, cpu_usage_str, mem_usage_str, degree, aff_str))
    {
        float peak_cpu, sum_cpu;
        float peak_mem, sum_mem;
        ResourceTS cpu_usage = retrieveResourceTS(cpu_usage_str, peak_cpu, sum_cpu);
        ResourceTS mem_usage = retrieveResourceTS(mem_usage_str, peak_mem, sum_mem);

        if( (cpu_usage.size() != size_series) or (mem_usage.size() != size_series))
        {
            // Problem in the resource usage of the application!
            std::string s = "Wrong size of resource usage for application " + app_id + ": found sizes ";
            s += std::to_string(cpu_usage.size()) + " and " + std::to_string(mem_usage.size());
            throw std::runtime_error(s);
        }

        // Make sure the replicas can be allocated to bins
        if ( (peak_cpu <= bin_cpu_capacity) and (peak_mem <= bin_mem_capacity) )
        {
            // Retrieve the map of affinities from the affinity string
            AffinityMap aff_map_out = constructAffinitiyMap(aff_str);

            // Update the in map of other items
            for(auto pair : aff_map_out)
            {
                auto it = maps_in.find(pair.first);
                if (it != maps_in.end())
                {
                    maps_in[pair.first][app_id] = pair.second;
                }
                else
                {
                    maps_in[pair.first] = AffinityMap();
                    maps_in[pair.first][app_id] = pair.second;
                }
            }

            app_list.push_back(new ApplicationTS(app_id, internal_id, nb_rep, size_series,
                                    cpu_usage, mem_usage,
                                    peak_cpu, peak_mem,
                                    degree, aff_map_out));
            internal_id++;

            // Update some counters
            for (int i = 0; i< size_series; ++i)
            {
                sum_cpu_TS[i] += nb_rep * cpu_usage[i];
                sum_mem_TS[i] += nb_rep * mem_usage[i];
            }
            total_sum_cpu_mem += nb_rep * (sum_cpu + sum_mem);
            total_replicas += nb_rep;
        }
        else
        {
            // Else a replica does not fit in a bin, drop the application
            to_remove.push_back(app_id);
        }
    }
    // Remove the filtered out apps from the affinity map
    // of remaining apps
    for (ApplicationTS* app : app_list)
    {
        app->setAffinityInMap(maps_in[app->getId()]);
        app->removeAppsAffinity(to_remove);
        app->setParams(sum_cpu_TS, sum_mem_TS, total_sum_cpu_mem,
                       total_replicas, bin_cpu_capacity, bin_mem_capacity);
    }
}

InstanceTS::~InstanceTS()
{
    for (ApplicationTS* app : app_list)
    {
        if (app != nullptr)
        {
            delete app;
        }
    }
}

const std::string& InstanceTS::getId() const
{
    return id;
}

const int InstanceTS::getBinCPUCapacity() const
{
    return bin_cpu_capacity;
}

const int InstanceTS::getBinMemCapacity() const
{
    return bin_mem_capacity;
}

const AppListTS& InstanceTS::getApps() const
{
    return app_list;
}

const size_t InstanceTS::getTSLength() const
{
    return TS_size;
}

const int InstanceTS::getTotalReplicas() const
{
    return total_replicas;
}

const ResourceTS& InstanceTS::getSumCPUTS() const
{
    return sum_cpu_TS;
}

const ResourceTS& InstanceTS::getSumMemTS() const
{
    return sum_mem_TS;
}


ResourceTS retrieveResourceTS(std::string resource_str, float &peak, float &sum)
{
    ResourceTS vect;
    peak = 0.0;
    sum = 0.0;
    float val;
    std::string str_val;
    std::istringstream splitStream(resource_str);

    while(std::getline(splitStream, str_val, ','))
    {
        str_val.erase(std::remove(str_val.begin(), str_val.end(), ' '), str_val.end());
        val = std::stof(str_val);
        vect.push_back(val);

        if (val > peak)
        {
            peak = val;
        }
        sum += val;

    }
    return vect;
}

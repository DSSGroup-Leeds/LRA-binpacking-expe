#ifndef INSTANCE_HPP
#define INSTANCE_HPP

#include "application.hpp"


void my_trim(std::string& s);
AffinityMap constructAffinitiyMap(std::string& aff_str);


class Instance2D
{
public:
    Instance2D(std::string id, int bin_cpu_capacity, int bin_memory_capacity,
               std::string& filename);

    virtual ~Instance2D();

    const std::string& getId() const;
    const int getBinCPUCapacity() const;
    const int getBinMemCapacity() const;
    const AppList2D& getApps() const;
    //const float getLambda() const;

    const int getSumCPU() const;
    const int getSumMem() const;
    const int getTotalReplicas() const;

private:
    std::string id;       // The instance id
    int bin_cpu_capacity; // The bin capacity for cpu requirements
    int bin_mem_capacity; // The bin capacity for memory requirements
    AppList2D app_list; // The list of Application2D of this instance

    int sum_mem;        // Total mem required by all replicas of apps
    int sum_cpu;        // Total cpu required by all replicas of apps
    int total_replicas;
};



class InstanceTS
{
public:
    InstanceTS(std::string id, int bin_cpu_capacity, int bin_mem_capacity,
               std::string& filename, size_t size_series);

    virtual ~InstanceTS();

    const std::string& getId() const;
    const int getBinCPUCapacity() const;
    const int getBinMemCapacity() const;
    const AppListTS& getApps() const;
    const size_t getTSLength() const;

    const int getTotalReplicas() const;
    const ResourceTS& getSumCPUTS() const;
    const ResourceTS& getSumMemTS() const;

private:
    std::string id;
    int bin_cpu_capacity;
    int bin_mem_capacity;
    AppListTS app_list;
    size_t TS_size;

    int total_replicas;
    ResourceTS sum_cpu_TS; // Sum of all applications cpu usage
    ResourceTS sum_mem_TS; // Sum of all applications memory usage
};

ResourceTS retrieveResourceTS(std::string resource_str, float &peak, float &sum);


#endif // INSTANCE_HPP

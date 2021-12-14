#ifndef APPLICATION_HPP
#define APPLICATION_HPP

#include <vector>
#include <string>
#include <unordered_map>


class Application2D;
class ApplicationTS;

using AppList2D = std::vector<Application2D*>;

using AffinityMap = std::unordered_map<std::string, int>;

using AppListTS = std::vector<ApplicationTS*>;
using ResourceTS = std::vector<float>; // A time series of resource consumption



class Application2D
{
public:
    Application2D(std::string& app_id, int internal_id,
                  int nb_replicas, int nb_cpus, int nb_memory,
                  int affinity_degree, AffinityMap& affinities);

    const std::string& getId() const;
    const int getInternalId() const;
    const int getNbReplicas() const;
    const int getCPUSize() const;
    const int getMemorySize() const;
    const float getNormalizedCPU() const;
    const float getNormalizedMemory() const;
    std::string toString(bool full = false) const;

    const int getOutDegree() const;
    const int getTotalDegree() const;
    const AffinityMap& getAffinityOutMap() const;
    const AffinityMap& getAffinityInMap() const;

    void removeAppsAffinity(std::vector<std::string>& to_remove);
    void setAffinityInMap(AffinityMap& affinities_in);

    virtual void setParams(float sum_cpu, float sum_mem, int total_replicas,
                   int bin_cpu_cap, int bin_mem_cap);
    const float getSurrogate() const;
    const float getExtSum() const;
    const float getAvgSize() const;
    const float getMaxSize() const;
    const float getAvgExpoSize() const;

    void setMeasure(float measure);
    const float getMeasure() const;

    void setFullyPacked(bool val);
    const bool isFullyPacked() const;

protected:
    std::string id;      // id of the application
    int internal_id;     // 0-based integer id
    int nb_replicas;     // number of replicas
    int nb_cpus;         // cpu requirement for each replica
    int nb_memory;     // memory requirement for each replica
    float norm_cpus;   // = nb_cpus / bin_cpu_capacity
    float norm_memory; // = nb_memory / bin_memory_capacity
    bool fully_packed;

    AffinityMap affinity_out_map; // map of affinity value pairs (app_b, k) of this item
        // Meaning that this item tolerates at most k replicas of app_b in the same bin
    AffinityMap affinity_in_map;  // map of affinity value pairs (app_b, k) from other items to this one
        // Meaning that at most k replicas of this item are tolerated by app_b in the same bin
    int affinity_out_degree; // size of the affinity out map
    int affinity_total_degree;// total number of neighbors (either in or out) <= (in_degree + out_degree)

    float avg_size;
    float max_size;
    float surrogate_size;
    float ext_sum_size;
    float avg_expo_size;

    float measure; // Placeholder for a measure value
};

Application2D* getApp2D(const AppList2D& list, const std::string& app_id);

bool application2D_comparator_total_degree_decreasing(Application2D* appa, Application2D* appb);
bool application2D_comparator_CPU_decreasing(Application2D* appa, Application2D* appb);

bool application2D_comparator_max_size_decreasing(Application2D* appa, Application2D* appb);
bool application2D_comparator_avg_size_decreasing(Application2D* appa, Application2D* appb);
bool application2D_comparator_surrogate_size_decreasing(Application2D* appa, Application2D* appb);
bool application2D_comparator_extsum_size_decreasing(Application2D* appa, Application2D* appb);
bool application2D_comparator_avgexpo_size_decreasing(Application2D* appa, Application2D* appb);

bool application2D_comparator_measure_increasing(Application2D* appa, Application2D* appb);
bool application2D_comparator_measure_decreasing(Application2D* appa, Application2D* appb);
void bubble_apps2D_down(AppList2D::iterator first, AppList2D::iterator last, bool comp (Application2D*, Application2D*));
void bubble_apps2D_up(AppList2D::iterator first, AppList2D::iterator last, bool comp (Application2D*, Application2D*));




class ApplicationTS : public Application2D
{
public:
    ApplicationTS(std::string& app_id, int internal_id,
                  int nb_replicas, size_t size_TS,
                  ResourceTS& cpu_usage, ResourceTS& mem_usage,
                  float peak_cpu, float peak_mem,
                  int affinity_degree, AffinityMap& affinities);

    const ResourceTS& getCpuUsage() const;
    const ResourceTS& getMemUsage() const;
    const ResourceTS& getNormCpuUsage() const;
    const ResourceTS& getNormMemUsage() const;


    void setParams(ResourceTS& sum_cpu, ResourceTS& sum_mem,
                   float total_sum_cpu_mem,
                   int total_replicas,
                   int bin_cpu_cap, int bin_mem_cap);

private:
    size_t TS_size;           // The size of the time series
    ResourceTS cpu_usage;     // Time series of cpu usage
    ResourceTS mem_usage;     // Time series of memory usage
    ResourceTS norm_cpus_TS;     // Time series of normalised cpu usage
    ResourceTS norm_memory_TS;   // Time series of normalised memory usage
    float peak_cpu;           // Max requirement of cpu usage
    float peak_mem;           // MAx requirement of memory usage
};

void bubble_appsTS_down(AppListTS::iterator first, AppListTS::iterator last, bool comp (Application2D*, Application2D*));
void bubble_appsTS_up(AppListTS::iterator first, AppListTS::iterator last, bool comp (Application2D*, Application2D*));

#endif // APPLICATION_HPP

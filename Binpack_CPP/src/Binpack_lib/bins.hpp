#ifndef BINS_HPP
#define BINS_HPP

#include "application.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <iterator>

class Bin2D;
class BinTS;

using BinList2D = std::vector<Bin2D*>;
using BinListTS = std::vector<BinTS*>;

using AllocMap = std::unordered_map<std::string, std::vector<int>>;
using ConflictMap = std::unordered_map<std::string, int>;


class Bin2D
{
public:
    Bin2D(int id, int max_cpu_capacity, int max_mem_capacity);
    Bin2D(const Bin2D& other) = default; // Copy ctor

    const int getId() const;
    const int getMaxCPUCap() const;
    const int getAvailableCPUCap() const;
    const int getMaxMemCap() const;
    const int getAvailableMemCap() const;

    const AllocMap& getAllocMap() const;
    const ConflictMap& getConflictMap() const;

    void addItem(Application2D* app, int replica_id);
    bool doesItemFit(int size_cpu, int size_mem) const;

    void printAlloc() const;

    //const int getAffinityValue(std::string& app_id) const;
    bool isAffinityCompliant(Application2D* app) const;

    // Updates the conflict_map with new
    // conflict/affinity values from this app
    // Must be called upon adding a new app in the bin
    void addNewConflict(Application2D* app);

    void setMeasure(float measure);
    const float getMeasure() const;

protected:
    const int id;
    const int max_cpu_capacity;
    const int max_mem_capacity;
    int available_cpu_capacity;
    int available_mem_capacity;

    // Maps an application id to a vector of replica id allocated to this bin
    AllocMap alloc_map;

    // Maps an application id to the number of replicas of this application
    // tolerated by the apps already packed in this bin
    ConflictMap conflict_map;

    float measure; // Placeholder for a measure value
};

bool bin2D_comparator_measure_increasing(Bin2D* bina, Bin2D* binb);
bool bin2D_comparator_measure_decreasing(Bin2D* bina, Bin2D* binb);

void bubble_bin_up(BinList2D::iterator first, BinList2D::iterator last, bool comp(Bin2D*, Bin2D*));
void bubble_bin_down(BinList2D::iterator first, BinList2D::iterator last, bool comp(Bin2D*, Bin2D*));

class BinTS : public Bin2D
{
public:
    BinTS(int id, int max_cpu_capacity, int max_mem_capacity, size_t size_TS);
    BinTS(const BinTS& other) = default; // Copy ctor

    void addItem(ApplicationTS* app, int replica_id);
    bool doesItemFit(ApplicationTS* app) const;

    const ResourceTS& getAvailableCPUCaps() const;
    const ResourceTS& getAvailableMemCaps() const;

    const float getTotalResidualCPU() const;
    const float getTotalResidualMem() const;
private:
    ResourceTS available_cpu_capacity;
    ResourceTS available_mem_capacity;
    size_t size_TS;
    float total_residual_cpu;
    float total_residual_mem;
};

void bubble_bin_up(BinListTS::iterator first, BinListTS::iterator last, bool comp(Bin2D*, Bin2D*));
void bubble_bin_down(BinListTS::iterator first, BinListTS::iterator last, bool comp(Bin2D*, Bin2D*));

#endif // BINS_HPP

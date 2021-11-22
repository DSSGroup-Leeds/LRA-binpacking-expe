#ifndef ALGOSTS_HPP
#define ALGOSTS_HPP

#include "application.hpp"
#include "instance.hpp"
#include "bins.hpp"

// Base class of AlgoFit tailored for TS bin packing
// With placeholder functions to sort the items, sort the bins
// and determine whether an items can be placed in a bin
class AlgoFitTS
{
public:
    AlgoFitTS(const InstanceTS &instance);
    virtual ~AlgoFitTS();

    bool isSolved() const;
    int getSolution() const;
    const BinListTS& getBins() const;
    BinListTS getBinsCopy() const;
    const AppListTS& getApps() const;
    const int getBinCPUCapacity() const;
    const int getBinMemCapacity() const;
    const std::string& getInstanceName() const;

    void setSolution(BinListTS& bins);
    void clearSolution();

    int solveInstance(int hint_nb_bins = 0);
    int solvePerBatch(int batch_size, int hint_nb_bins = 0);

private:
    virtual void createNewBin(); // Open a new empty bin
    virtual void allocateBatch(AppListTS::iterator first_app, AppListTS::iterator end_batch);

    // These are the 4 methods each variant of the Fit algo should implement
    virtual void sortBins() = 0;
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it) = 0;
    virtual bool checkItemToBin(ApplicationTS* app, BinTS* bin) const = 0;
    virtual void addItemToBin(ApplicationTS* app, int replica_id, BinTS* bin) = 0;

protected:
    std::string instance_name;
    size_t size_TS;
    int bin_cpu_capacity;
    int bin_mem_capacity;
    int total_replicas;
    const ResourceTS& sum_cpu_TS;
    const ResourceTS& sum_mem_TS;
    int next_bin_index;
    int curr_bin_index;
    AppListTS apps;
    BinListTS bins;
    bool solved;
};

// Creator of AlgoFitTS variant w.r.t. given algo_name
AlgoFitTS* createAlgoTS(const std::string& algo_name, const InstanceTS &instance);


/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ First Fit Affinity *********/
class AlgoTSFF : public AlgoFitTS
{
public:
    AlgoTSFF(const InstanceTS &instance);
protected:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
    virtual void sortBins();
    virtual bool checkItemToBin(ApplicationTS* app, BinTS* bin) const;
    virtual void addItemToBin(ApplicationTS* app, int replica_id, BinTS* bin);
};


/************ First Fit Decreasing Degree Affinity *********/
class AlgoTSFFDDegree : public AlgoTSFF
{
public:
    AlgoTSFFDDegree(const InstanceTS &instance);
private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
};

/************ First Fit Decreasing Average Affinity *********/
class AlgoTSFFDAvg : public AlgoTSFF
{
public:
    AlgoTSFFDAvg(const InstanceTS &instance);
private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
};

/************ First Fit Decreasing Max Affinity *********/
class AlgoTSFFDMax : public AlgoTSFF
{
public:
    AlgoTSFFDMax(const InstanceTS &instance);
private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
};

/************ First Fit Decreasing Average with Exponential Weights Affinity *********/
class AlgoTSFFDAvgExpo : public AlgoTSFF
{
public:
    AlgoTSFFDAvgExpo(const InstanceTS &instance);
private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
};

/************ First Fit Decreasing Surrogate Affinity *********/
class AlgoTSFFDSurrogate : public AlgoTSFF
{
public:
    AlgoTSFFDSurrogate(const InstanceTS &instance);

protected:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
};

/************ First Fit Decreasing Extended Sum Affinity *********/
class AlgoTSFFDExtendedSum : public AlgoTSFF
{
public:
    AlgoTSFFDExtendedSum(const InstanceTS &instance);

protected:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
};



/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ Best Fit Decreasing Average Affinity *********/
class AlgoTSBFDAvg : public AlgoFitTS
{
public:
    AlgoTSBFDAvg(const InstanceTS &instance);
private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
    virtual void sortBins();
    virtual bool checkItemToBin(ApplicationTS* app, BinTS* bin) const;
    virtual void addItemToBin(ApplicationTS* app, int replica_id, BinTS* bin);

    virtual void updateBinMeasure(BinTS* bin);
};

/************ Best Fit Decreasing Max Affinity *********/
class AlgoTSBFDMax : public AlgoTSBFDAvg
{
public:
    AlgoTSBFDMax(const InstanceTS &instance);
private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
    virtual void updateBinMeasure(BinTS* bin);
};


/************ Best Fit Decreasing Average with Exponential Weights Affinity *********/
class AlgoTSBFDAvgExpo : public AlgoTSBFDAvg
{
public:
    AlgoTSBFDAvgExpo(const InstanceTS &instance);
private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
    virtual void createNewBin();
    virtual void addItemToBin(ApplicationTS *app, int replica_id, BinTS *bin);
    virtual void updateBinMeasure(BinTS* bin);
protected:
    float total_residual_cpu;
    float total_residual_mem;
};



/************ Best Fit Decreasing Surrogate Affinity *********/
class AlgoTSBFDSurrogate : public AlgoTSBFDAvgExpo
{
public:
    AlgoTSBFDSurrogate(const InstanceTS &instance);

private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
    virtual void updateBinMeasure(BinTS* bin);
};

/************ Best Fit Decreasing Extended Sum Affinity *********/
class AlgoTSBFDExtendedSum : public AlgoTSBFDAvgExpo
{
public:
    AlgoTSBFDExtendedSum(const InstanceTS &instance);

private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
    virtual void createNewBin();
    virtual void addItemToBin(ApplicationTS *app, int replica_id, BinTS *bin);
    virtual void updateBinMeasure(BinTS* bin);

protected:
    ResourceTS sum_residual_cpu; // Sum of residual capacity
    ResourceTS sum_residual_mem; // of all bins for each time step
};




/* ================================================ */
/* ================================================ */
/* ================================================ */
/********* Bin Centric FFD DotProduct ***************/
class AlgoTSBinFFDDotProduct : public AlgoTSFF
{
public:
    AlgoTSBinFFDDotProduct(const InstanceTS &instance);

protected:
    virtual void allocateBatch(AppListTS::iterator first_app, AppListTS::iterator end_batch);
    virtual bool isBinFilled(BinTS* bin);
    virtual void computeMeasures(AppListTS::iterator start_list, AppListTS::iterator end_list, BinTS* bin);
};


/********* Bin Centric FFD L2Norm ***************/
class AlgoTSBinFFDL2Norm : public AlgoTSBinFFDDotProduct
{
public:
    AlgoTSBinFFDL2Norm(const InstanceTS &instance);

protected:
    virtual void computeMeasures(AppListTS::iterator start_list, AppListTS::iterator end_list, BinTS* bin);
};


/********* Bin Centric FFD Fitness ***************/
class AlgoTSBinFFDFitness : public AlgoTSBinFFDDotProduct
{
public:
    AlgoTSBinFFDFitness(const InstanceTS &instance);

protected:
    virtual void computeMeasures(AppListTS::iterator start_list, AppListTS::iterator end_list, BinTS* bin);
};



#endif // ALGOSTS_HPP
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
    virtual void sortBins();
    virtual void createNewBin();
    virtual void addItemToBin(ApplicationTS *app, int replica_id, BinTS *bin);
    virtual void updateBinMeasure(BinTS* bin);
protected:
    ResourceTS sum_residual_cpu; // Sum of residual capacity
    ResourceTS sum_residual_mem; // of all bins for each time step
};



/************ Best Fit Decreasing Surrogate Affinity *********/
class AlgoTSBFDSurrogate : public AlgoTSBFDAvgExpo
{
public:
    AlgoTSBFDSurrogate(const InstanceTS &instance);

private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
    virtual void sortBins();
    virtual void updateBinMeasure(BinTS* bin);
};

/************ Best Fit Decreasing Extended Sum Affinity *********/
class AlgoTSBFDExtendedSum : public AlgoTSBFDAvgExpo
{
public:
    AlgoTSBFDExtendedSum(const InstanceTS &instance);

private:
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
    virtual void sortBins();
    //virtual void createNewBin();
    //virtual void addItemToBin(ApplicationTS *app, int replica_id, BinTS *bin);
    virtual void updateBinMeasure(BinTS* bin);
};



/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ Worst Fit Decreasing Avg Affinity *********/
class AlgoTSWFDAvg : public AlgoTSBFDAvg
{
public:
    AlgoTSWFDAvg(const InstanceTS &instance);
private:
    virtual void sortBins();
};

/************ Worst Fit Decreasing Max Affinity *********/
class AlgoTSWFDMax : public AlgoTSBFDMax
{
public:
    AlgoTSWFDMax(const InstanceTS &instance);
private:
    virtual void sortBins();
};

/************ Worst Fit Decreasing AvgExpo Affinity *********/
class AlgoTSWFDAvgExpo : public AlgoTSBFDAvgExpo
{
public:
    AlgoTSWFDAvgExpo(const InstanceTS &instance);
private:
    virtual void sortBins();
};

/************ Worst Fit Decreasing Surrogate Affinity *********/
class AlgoTSWFDSurrogate : public AlgoTSBFDSurrogate
{
public:
    AlgoTSWFDSurrogate(const InstanceTS &instance);
private:
    virtual void sortBins();
};

/************ Worst Fit Decreasing ExtendedSum Affinity *********/
class AlgoTSWFDExtendedSum : public AlgoTSBFDExtendedSum
{
public:
    AlgoTSWFDExtendedSum(const InstanceTS &instance);
private:
    virtual void sortBins();
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
    virtual BinTS* createNewBinRet();
};

/********* Bin Centric FFD DotDivision ***************/
class AlgoTSBinFFDDotDivision : public AlgoTSBinFFDDotProduct
{
public:
    AlgoTSBinFFDDotDivision(const InstanceTS &instance);

protected:
    virtual void computeMeasures(AppListTS::iterator start_list, AppListTS::iterator end_list,  BinTS* bin);
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

    virtual BinTS* createNewBinRet();
    virtual void addItemToBin(ApplicationTS *app, int replica_id, BinTS* bin);
    ResourceTS sum_residual_cpu; // Sum of residual capacity
    ResourceTS sum_residual_mem; // of all bins for each time step
};





/* ================================================ */
/* ================================================ */
/* ================================================ */
/*********** Spread replicas Worst Fit Avg **********/
class AlgoTSSpreadWFDAvg : public AlgoFitTS
{
public:
    AlgoTSSpreadWFDAvg(const InstanceTS &instance);

    int solveInstanceSpread(int LB_bins, int UB_bins);
    bool trySolve(int nb_bins); // Try to find a solution with the given bins
private:

    virtual void createBins(int nb_bins);
    virtual void updateBinMeasure(BinTS* bin);
    virtual void updateBinMeasures();          // If all bins need to be updated

    virtual void allocateBatch(AppListTS::iterator first_app, AppListTS::iterator end_batch);

    virtual void sortBins();
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
    virtual bool checkItemToBin(ApplicationTS* app, BinTS* bin) const;
    virtual void addItemToBin(ApplicationTS* app, int replica_id, BinTS* bin);
};

/*********** Spread replicas Worst Fit Max **********/
class AlgoTSSpreadWFDMax : public AlgoTSSpreadWFDAvg
{
public:
    AlgoTSSpreadWFDMax(const InstanceTS &instance);

private:
    virtual void updateBinMeasure(BinTS* bin);
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
};

/*********** Spread replicas Worst Fit Surrogate **********/
class AlgoTSSpreadWFDSurrogate : public AlgoTSSpreadWFDAvg
{
public:
    AlgoTSSpreadWFDSurrogate(const InstanceTS &instance);

private:
    virtual void createBins(int nb_bins);
    virtual void updateBinMeasure(BinTS* bin);
    virtual void updateBinMeasures();
    virtual void addItemToBin(ApplicationTS *app, int replica_id, BinTS *bin);
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);

protected:
    ResourceTS sum_residual_cpu;
    ResourceTS sum_residual_mem;
};


/*********** Spread replicas Worst Fit AvgExpo **********/
/*class AlgoTSSpreadWFDAvgExpo : public AlgoTSSpreadWFDSurrogate
{
public:
    AlgoTSSpreadWFDAvgExpo(const InstanceTS &instance);

private:
    //virtual void createBins(int nb_bins);
    virtual void updateBinMeasure(BinTS* bin);
    virtual void updateBinMeasures();
    //virtual void addItemToBin(ApplicationTS *app, int replica_id, BinTS *bin);
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
};*/



/*********** Spread replicas Worst Fit Extended Sum **********/
/*class AlgoTSSpreadWFDExtendedSum : public AlgoTSSpreadWFDAvgExpo
{
public:
    AlgoTSSpreadWFDExtendedSum(const InstanceTS &instance);

private:
    virtual void updateBinMeasure(BinTS* bin);
    virtual void updateBinMeasures();
    virtual void sortApps(AppListTS::iterator first_app, AppListTS::iterator end_it);
};*/


/* ================================================ */
/* ================================================ */
/* ================================================ */
/**** A variant of SpreadWFD algorithms *************/
class AlgoTSRefineWFDAvg : public AlgoTSSpreadWFDAvg
{
public:
    AlgoTSRefineWFDAvg(const InstanceTS &instance, const float ratio);

protected:
    virtual int solveInstanceSpread(int LB_bins, int UB_bins);

private:
    float ratio_refinement;
};




AlgoTSSpreadWFDAvg* createSpreadAlgo(const std::string &algo_name, const InstanceTS &instance);


#endif // ALGOSTS_HPP

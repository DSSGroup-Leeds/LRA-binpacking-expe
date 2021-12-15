#ifndef ALGOS2D_HPP
#define ALGOS2D_HPP

#include "application.hpp"
#include "instance.hpp"
#include "bins.hpp"

// Base class of AlgoFit tailored for 2D bin packing
// With placeholder functions to sort the items, sort the bins
// and determine whether an item can be placed in a bin
class AlgoFit2D
{
public:
    AlgoFit2D(const Instance2D &instance);
    virtual ~AlgoFit2D();

    bool isSolved() const;
    int getSolution() const;
    const BinList2D& getBins() const;
    BinList2D getBinsCopy() const;
    const AppList2D& getApps() const;
    const int getBinCPUCapacity() const;
    const int getBinMemCapacity() const;
    const std::string& getInstanceName() const;

    void setSolution(BinList2D& bins);
    void clearSolution();

    int solveInstance(int hint_nb_bins = 0);
    int solvePerBatch(int batch_size, int hint_nb_bins = 0);

private:
    virtual void createNewBin(); // Open a new empty bin
    virtual void allocateBatch(AppList2D::iterator first_app, AppList2D::iterator end_batch);

    // These are the methods each variant of the Fit algo should implement
    virtual void sortBins() = 0;
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it) = 0;
    virtual bool checkItemToBin(Application2D* app, Bin2D* bin) const = 0;
    virtual void addItemToBin(Application2D* app, int replica_id, Bin2D* bin) = 0;

protected:
    std::string instance_name;
    int bin_cpu_capacity;
    int bin_mem_capacity;
    int total_replicas;
    int sum_cpu;
    int sum_mem;
    float norm_sum_cpu;
    float norm_sum_mem;
    int next_bin_index;
    int curr_bin_index;
    AppList2D apps;
    BinList2D bins;
    bool solved;
};

// Creator of AlgoFit2D variant w.r.t. given algo_name
AlgoFit2D* createAlgo2D(const std::string &algo_name, const Instance2D &instance);

/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ First Fit Affinity *********/
class Algo2DFF : public AlgoFit2D
{
public: 
    Algo2DFF(const Instance2D &instance);
protected:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
    virtual void sortBins();
    virtual bool checkItemToBin(Application2D* app, Bin2D* bin) const;
    virtual void addItemToBin(Application2D* app, int replica_id, Bin2D* bin);
};


/************ First Fit Decreasing Degree Affinity *********/
class Algo2DFFDDegree : public Algo2DFF
{
public:
    Algo2DFFDDegree(const Instance2D &instance);
private:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
};

/************ First Fit Decreasing Average Affinity *********/
class Algo2DFFDAvg : public Algo2DFF
{
public: 
    Algo2DFFDAvg(const Instance2D &instance);
private:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
};

/************ First Fit Decreasing Average with Exponential Weights Affinity *********/
class Algo2DFFDAvgExpo : public Algo2DFF
{
public:
    Algo2DFFDAvgExpo(const Instance2D &instance);
private:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
};

/************ First Fit Decreasing Max Affinity *********/
class Algo2DFFDMax : public Algo2DFF
{
public: 
    Algo2DFFDMax(const Instance2D &instance);
private:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
};

/************ First Fit Decreasing Surrogate Affinity *********/
class Algo2DFFDSurrogate : public Algo2DFF
{
public: 
    Algo2DFFDSurrogate(const Instance2D &instance);

protected:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
};

/************ First Fit Decreasing Extended Sum Affinity *********/
class Algo2DFFDExtendedSum : public Algo2DFF
{
public:
    Algo2DFFDExtendedSum(const Instance2D &instance);

protected:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
};



/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ Best Fit Decreasing Average Affinity *********/
class Algo2DBFDAvg : public AlgoFit2D
{
public:
    Algo2DBFDAvg(const Instance2D &instance);
private:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
    virtual void sortBins();
    virtual bool checkItemToBin(Application2D* app, Bin2D* bin) const;
    virtual void addItemToBin(Application2D* app, int replica_id, Bin2D* bin);

    virtual void updateBinMeasure(Bin2D* bin);
};

/************ Best Fit Decreasing Max Affinity *********/
class Algo2DBFDMax : public Algo2DBFDAvg
{
public:
    Algo2DBFDMax(const Instance2D &instance);
private:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
    virtual void updateBinMeasure(Bin2D* bin);
};

/************ Best Fit Decreasing Average with Exponential Weights Affinity *********/
class Algo2DBFDAvgExpo : public Algo2DBFDAvg
{
public:
    Algo2DBFDAvgExpo(const Instance2D &instance);
private:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
    virtual void sortBins();
    virtual void createNewBin();
    virtual void addItemToBin(Application2D *app, int replica_id, Bin2D *bin);
    virtual void updateBinMeasure(Bin2D* bin);
protected:
    int total_residual_cpu;
    int total_residual_mem;
};

/************ Best Fit Decreasing Surrogate Affinity *********/
class Algo2DBFDSurrogate : public Algo2DBFDAvgExpo
{
public:
    Algo2DBFDSurrogate(const Instance2D &instance);

protected:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
    virtual void sortBins();
    virtual void updateBinMeasure(Bin2D* bin);
};

/************ Best Fit Decreasing Extended Sum Affinity *********/
class Algo2DBFDExtendedSum : public Algo2DBFDAvgExpo
{
public:
    Algo2DBFDExtendedSum(const Instance2D &instance);

protected:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
    virtual void sortBins();
    virtual void updateBinMeasure(Bin2D* bin);
};




/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ Worst Fit Decreasing Avg Affinity *********/
class Algo2DWFDAvg : public Algo2DBFDAvg
{
public:
    Algo2DWFDAvg(const Instance2D &instance);
private:
    virtual void sortBins();
};

/************ Worst Fit Decreasing Max Affinity *********/
class Algo2DWFDMax : public Algo2DBFDMax
{
public:
    Algo2DWFDMax(const Instance2D &instance);
private:
    virtual void sortBins();
};

/************ Worst Fit Decreasing AvgExpo Affinity *********/
class Algo2DWFDAvgExpo : public Algo2DBFDAvgExpo
{
public:
    Algo2DWFDAvgExpo(const Instance2D &instance);
private:
    virtual void sortBins();
};

/************ Worst Fit Decreasing Surrogate Affinity *********/
class Algo2DWFDSurrogate : public Algo2DBFDSurrogate
{
public:
    Algo2DWFDSurrogate(const Instance2D &instance);
private:
    virtual void sortBins();
};

/************ Worst Fit Decreasing ExtendedSum Affinity *********/
class Algo2DWFDExtendedSum : public Algo2DBFDExtendedSum
{
public:
    Algo2DWFDExtendedSum(const Instance2D &instance);
private:
    virtual void sortBins();
};



/* ================================================ */
/* ================================================ */
/* ================================================ */
/************ Medea Node Count Affinity *********/
class Algo2DNodeCount : public AlgoFit2D
{
public:
    Algo2DNodeCount(const Instance2D &instance);
protected:
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
    virtual void sortBins();
    virtual bool checkItemToBin(Application2D* app, Bin2D* bin) const;
    virtual void addItemToBin(Application2D* app, int replica_id, Bin2D* bin);

private:
    virtual void allocateBatch(AppList2D::iterator first_app, AppList2D::iterator end_batch);
};




/* ================================================ */
/* ================================================ */
/* ================================================ */
/********* Bin Centric FFD DotProduct ***************/
class Algo2DBinFFDDotProduct : public Algo2DFF
{
public:
    Algo2DBinFFDDotProduct(const Instance2D &instance);

private:
    virtual void allocateBatch(AppList2D::iterator first_app, AppList2D::iterator end_batch);
protected:
    virtual Bin2D* createNewBinRet();
    virtual bool isBinFilled(Bin2D* bin);
    virtual void computeMeasures(AppList2D::iterator start_list, AppList2D::iterator end_list, Bin2D* bin);
};

/********* Bin Centric FFD DotDivision ***************/
class Algo2DBinFFDDotDivision : public Algo2DBinFFDDotProduct
{
public:
    Algo2DBinFFDDotDivision(const Instance2D &instance);

protected:
    virtual void computeMeasures(AppList2D::iterator start_list, AppList2D::iterator end_list,  Bin2D* bin);
};

/********* Bin Centric FFD L2Norm ***************/
class Algo2DBinFFDL2Norm : public Algo2DBinFFDDotProduct
{
public:
    Algo2DBinFFDL2Norm(const Instance2D &instance);

protected:
    virtual void computeMeasures(AppList2D::iterator start_list, AppList2D::iterator end_list,  Bin2D* bin);
};

/********* Bin Centric FFD Fitness ***************/
class Algo2DBinFFDFitness : public Algo2DBinFFDDotProduct
{
public:
    Algo2DBinFFDFitness(const Instance2D &instance);

protected:
    virtual void computeMeasures(AppList2D::iterator start_list, AppList2D::iterator end_list,  Bin2D* bin);

    virtual Bin2D* createNewBinRet();
    virtual void addItemToBin(Application2D *app, int replica_id, Bin2D *bin);
    int total_residual_cpu;
    int total_residual_mem;
};


/* ================================================ */
/* ================================================ */
/* ================================================ */
/*********** Spread replicas Worst Fit Avg **********/
class Algo2DSpreadWFDAvg : public AlgoFit2D
{
public:
    Algo2DSpreadWFDAvg(const Instance2D &instance);

    int solveInstanceSpread(int LB_bins, int UB_bins);
private:
    bool trySolve(int nb_bins); // Try to find a solution with the given bins

    virtual void createBins(int nb_bins);
    virtual void updateBinMeasure(Bin2D* bin);
    virtual void updateBinMeasures();          // If all bins need to be updated

    virtual void allocateBatch(AppList2D::iterator first_app, AppList2D::iterator end_batch);

    virtual void sortBins();
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
    virtual bool checkItemToBin(Application2D* app, Bin2D* bin) const;
    virtual void addItemToBin(Application2D* app, int replica_id, Bin2D* bin);
};

/*********** Spread replicas Worst Fit Max **********/
class Algo2DSpreadWFDMax : public Algo2DSpreadWFDAvg
{
public:
    Algo2DSpreadWFDMax(const Instance2D &instance);

private:
    virtual void updateBinMeasure(Bin2D* bin);
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
};


/*********** Spread replicas Worst Fit AvgExpo **********/
class Algo2DSpreadWFDAvgExpo : public Algo2DSpreadWFDAvg
{
public:
    Algo2DSpreadWFDAvgExpo(const Instance2D &instance);

private:
    virtual void createBins(int nb_bins);
    virtual void updateBinMeasure(Bin2D* bin);
    virtual void updateBinMeasures();
    virtual void addItemToBin(Application2D *app, int replica_id, Bin2D *bin);
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);

protected:
    int total_residual_cpu;
    int total_residual_mem;
};

/*********** Spread replicas Worst Fit Surrogate **********/
class Algo2DSpreadWFDSurrogate : public Algo2DSpreadWFDAvgExpo
{
public:
    Algo2DSpreadWFDSurrogate(const Instance2D &instance);

private:
    virtual void updateBinMeasure(Bin2D* bin);
    virtual void updateBinMeasures();
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
};


/*********** Spread replicas Worst Fit Extended Sum **********/
class Algo2DSpreadWFDExtendedSum : public Algo2DSpreadWFDAvgExpo
{
public:
    Algo2DSpreadWFDExtendedSum(const Instance2D &instance);

private:
    virtual void updateBinMeasure(Bin2D* bin);
    virtual void updateBinMeasures();
    virtual void sortApps(AppList2D::iterator first_app, AppList2D::iterator end_it);
};

Algo2DSpreadWFDAvg* createSpreadAlgo(const std::string &algo_name, const Instance2D &instance);

#endif // ALGOS2D_HPP

 #include "lower_bounds.hpp"

#include <cmath>
#include <unordered_set>


int BPP2D_LBalpha_mem(const Instance2D & instance, const int alpha)
{
    if ( ((float)alpha) > ((float)instance.getBinMemCapacity() / 2.0) )
    {
        return 0;
    }

    float threshold1 = (float)instance.getBinMemCapacity() - ((float)alpha);
    float threshold2 = (float)instance.getBinMemCapacity() / 2.0;
    int nJ1 = 0;
    int nJ2 = 0;
    float sumJ2 = 0.0;
    float sumJ3 = 0.0;

    for (Application2D* app : instance.getApps())
    {
        if (app->getMemorySize() > threshold1)
        {
            nJ1 += app->getNbReplicas();
        }
        else if (app->getMemorySize() > threshold2)
        {
            nJ2 += app->getNbReplicas();
            sumJ2 += app->getMemorySize() * app->getNbReplicas();
        }
        else if (app->getMemorySize() >= alpha)
        {
            sumJ3 += app->getMemorySize() * app->getNbReplicas();
        }
    }

    int tmp = (int)std::ceil( (sumJ3 + sumJ2 - (nJ2 * instance.getBinMemCapacity())) / instance.getBinMemCapacity() );
    return nJ1 + nJ2 + std::max(0, tmp);
}

int BPP2D_LBalpha_cpu(const Instance2D & instance, const int alpha)
{
    if ( ((float)alpha) > ((float)instance.getBinCPUCapacity() / 2.0) )
    {
        return 0;
    }

    float threshold1 = (float)instance.getBinCPUCapacity() - ((float)alpha);
    float threshold2 = (float)instance.getBinCPUCapacity() / 2.0;
    int nJ1 = 0;
    int nJ2 = 0;
    float sumJ2 = 0.0;
    float sumJ3 = 0.0;

    for (Application2D* app : instance.getApps())
    {
        if (app->getCPUSize() > threshold1)
        {
            nJ1 += app->getNbReplicas();
        }
        else if (app->getCPUSize() > threshold2)
        {
            nJ2 += app->getNbReplicas();
            sumJ2 += app->getCPUSize() * app->getNbReplicas();
        }
        else if (app->getCPUSize() >= alpha)
        {
            sumJ3 += app->getCPUSize() * app->getNbReplicas();
        }
    }

    int tmp = (int)std::ceil( (sumJ3 + sumJ2 - (nJ2 * instance.getBinCPUCapacity())) / instance.getBinCPUCapacity() );
    return nJ1 + nJ2 + std::max(0, tmp);
}


int BPP2D_LBcpu(const Instance2D & instance)
{
    int LB_cpu = 0;
    std::unordered_set<int> alpha_set_cpu;

    for (Application2D* app : instance.getApps())
    {
        alpha_set_cpu.insert(app->getCPUSize());
    }
    for (int alpha : alpha_set_cpu)
    {
        int LB_alpha = BPP2D_LBalpha_cpu(instance, alpha);
        LB_cpu = std::max(LB_cpu, LB_alpha);
    }
    return LB_cpu;
}

int BPP2D_LBmem(const Instance2D & instance)
{
    int LB_mem = 0;
    std::unordered_set<int> alpha_set_mem;

    for (Application2D* app : instance.getApps())
    {
        alpha_set_mem.insert(app->getMemorySize());
    }
    for (int alpha : alpha_set_mem)
    {
        int LB_alpha = BPP2D_LBalpha_mem(instance, alpha);
        LB_mem = std::max(LB_mem, LB_alpha);
    }
    return LB_mem;
}

int BPP2D_LB(const Instance2D & instance)
{
    int LB_cpu = BPP2D_LBcpu(instance);
    int LB_mem = BPP2D_LBmem(instance);

    return std::max(LB_cpu, LB_mem);
}


void TS_LB(const InstanceTS & instance, int &LB_cpu, int &LB_mem)
{
    float max_cpu = 0.0;
    for (float f : instance.getSumCPUTS())
    {
        if (max_cpu < f)
        {
            max_cpu = f;
        }
    }

    float max_mem = 0.0;
    for (float f : instance.getSumMemTS())
    {
        if (max_mem < f)
        {
            max_mem = f;
        }
    }

    LB_cpu = std::ceil(max_cpu / instance.getBinCPUCapacity());
    LB_mem = std::ceil(max_mem / instance.getBinMemCapacity());
}

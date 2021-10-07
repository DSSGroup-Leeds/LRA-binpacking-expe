#ifndef LOWER_BOUNDS_HPP
#define LOWER_BOUNDS_HPP

#include <application.hpp>
#include <instance.hpp>

int BPP2D_LBalpha_cpu(const Instance2D & instance, const int alpha);
int BPP2D_LBalpha_mem(const Instance2D & instance, const int alpha);
int BPP2D_LBcpu(const Instance2D & instance);
int BPP2D_LBmem(const Instance2D & instance);
int BPP2D_LB(const Instance2D & instance);

void TS_LB(const InstanceTS & instance, int &LB_cpu, int &LB_mem);

#endif // LOWER_BOUNDS_HPP

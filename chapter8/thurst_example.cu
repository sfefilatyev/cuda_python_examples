/***
This script is an example of usign CUDA Thrust library.
***/

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

using namespace std;

int main(void)
{
    thrust::host_vector<int> v;
    v.push_back(1);
    v.push_back(2);
    v.push_back(3);
    v.push_back(4);

    for (int i = 0; i < v.size(); i++)
        cout << "v[" << i << "] == " << v[i] << endl;

    thrust::device_vector<int> v_gpu = v;
    v_gpu.push_back(5);

    for (int i = 0; i < v_gpu.size(); i++)
        std::cout << "v_gpu[" << i << "] == " << v_gpu[i] << std::endl;

    return 0;
}

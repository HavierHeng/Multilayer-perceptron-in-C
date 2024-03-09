#include "laf.h"

int main( int argc, char* argv[] ) {
    unsigned int x = 0x000bffff;
    unsigned int w = 0x000bffff;
    unsigned int res = 0;
    unsigned int * p;
    p = (unsigned int*)0x3FFC;

    // Multiplier: EXP(x_LV + w_LV) 
    res = x + w;
    EXP(res);

    res <<= 2;
    AF(res);
    LOG(res);

    *p = res;

    while(1);
    return 0;
}

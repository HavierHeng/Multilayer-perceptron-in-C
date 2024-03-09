#include "laf_sw.h"

// LAF Implementations are based off original paper by Prof Miroslav Skrbek

/* --------------- Helper Function --------------- */
float absolute(float x) {
    return (x < 0.0) ? -x : x;
}

int intPart(float x) {
    return absolute((int)x);
}

float fracPart(float x){
    return absolute(x - (int)x);
}

int signum(float x) {
    return (x > 0) - (x < 0);
}

int leftmostBit(int n){
    if (n == 0) {
        // Case when n is 0
        return -1;  // No set bits
    }
    // Bit smearing technique
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);

    // Leftmost bit is now alone in its column
    // Extract position of leftmost 1 value
    int position = 0;

    while ((n & 1) == 1) {
        position++;
        n >>= 1;  // Right shift by 1 position
    }

    return position - 1;
}

float power(float x, float n) {
    // Binary Exponential Algorithm
    // Use bitshifting unless you have non-integer exponents and base
    if (n == 0.0) {
        return 1.0;  // x^0 is always 1
    }

    float result = 1.0;
    float base = x;
    int exponent;

    if (n < 0.0) {
        base = 1.0 / x;
        n = -n;
    }

    while (n > 0) {
        if ((int)n % 2 == 1) {
            result *= base;
        }
        base *= base;
        n /= 2.0;
    }

    return result;
}

/* --------------- Simple Blocks --------------- */

float EXP(float val) {
    // 2^x
    int intP = intPart(val);
    float fracP = fracPart(val);
    return (1 << intP) * (1 + fracP);
}

float LOG(float val) {
    // log base 2(x), valid for x from [1, inf)
    if (val < 1) return 0;
    int intP = leftmostBit(val);
    return intP + (val / (1 << intP)) - 1;
}
    
/* --------------- Activation Functions --------------- */

float AF(float val){
    // Sigmoid/tanh Activation Function
    int sgn = signum(val);
    return sgn * ((1.0 / (1 << intPart(absolute(val)))) * ((fracPart(absolute(val)) / 2.0) - 1.0) + 1);
}

float AFR(float val){
    // Gaussian Activation Function
    return (1.0 / (1 << intPart(absolute(val)))) * (1 - fracPart(absolute(val)) / 2.0);
}

/* --------------- Complex Blocks --------------- */

#define WIDTH 14 // Width of bit grid, max is 14 assuming unsigned int (32 bits), and WIDTH has to be even
 
float MUL(float x, float y) {
    return signum(x) * signum(y) * EXP(LOG((1 << WIDTH) * absolute(x)) + LOG((1 << WIDTH) * absolute(y))) / (1 << (2 * WIDTH));
}

float SQR(float val) {
    // Square, x in range of (-1, 1)
    return EXP(2 * LOG((1 << WIDTH) * absolute(val))) / (1 << (2 * WIDTH));
}

float SQRT(float val) {  
    // Square Root, x in range of (0, 1)
    return EXP(0.5 * LOG((1 << WIDTH) * val)) / (power(2, 0.5 * WIDTH));
}


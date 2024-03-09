#ifndef LAF  // Only allow either laf.h or laf_sw.h
#define LAF

// Simple Blocks
float EXP(float val);  // 2^x
float LOG(float val);  // log base 2(x)

// Activation Functions
float AF(float val);  // Sigmoid/tanh Activation Function
float AFR(float val);  // Gaussian Activation Function

// Complex Blocks
float MUL(float x, float y);  // Multiply
float SQR(float val);  // Square 
float SQRT(float val);  // Square Root

#endif

# Multilayer-perceptron-in-C
A project that implements a free standing multilayer perceptron in C (MLP) without use of any stdlibs, only linear approximated math. With reference to [Feasibility of a Neural Network with Linearly Approximated Functions on Zynq FPGA by Skrbek, M., Kubalík, P.	](10.1109/ICECS202256217.2022.9970813 )

## Motivation/Objective

The original objective of this project was to implement custom extension instructions for RISC-V that use a linear aproximation of non-linear functions used by machine learining. However to perform a proper benchmark, there was a need to create a ML neural network in software using C first, given the nature of the project ensures that there will be no software support, only raw RISC-V assembly or a RISC-V toolchain.

Only then can the model be taught on input data and test it on "standard" RISC-V implementation to compare against the hardware implementation in an FPGA.

## Initial problems faced
### Lack of standard libaries in freestanding environment
As the custom RISC V components are expected to not have any kernel or operating system, just raw bytecode into a BRAM, the use of standard libraries are frowned upon as some states like `printf` result in a trap to the kernel. 

However, in the final code, the feature to use the C stdlibs can be toggled by a `DEBUG` macro. This was important to validate the output of the software on standard programming environments (tested on [C compiler explorer](https://godbolt.org/)).

### Need for customizability
Another minor concern was that standard instructions like MUL could be disabled based on the RISC V extension spec (R32VIM extension). While the actual hardware implemented in an FPGA could toggle it on and off, in the case that it could not, a linearly approximated model would be used as a stand in.


## Testing and experimentation

### Software Evaluation of LAF functions 
The LAF functions were evaluated in Python first to determine their errors as compared to the standard implementations. As compared to the original paper, this seems to have yielded different values due to either architectural (precision of system) or implementation differences (such as how Numpy approximates certain calculations), as well as the testing ranges. The code used to generate the evaluation stats can be found in this [Python Notebook]('XOR%20Problem%20and%20LAF%20Graphs.ipynb').

The error calculations are performed under the following assumptions. 
1) All values are 32 bits - Use of `float` and `int`.
2) For LAF functions, such as complex function blocks, they are composed from other LAF blocks as much as possible. Complex function blocks use 16 bits as the calculation space.
3) Errors are calculated based off 1000 uniformly spaced samples from the X Lower to X Upper bounds.

<div align="center">
  <img src="../sources/LAF-EXP.png" width="45%" />
  <img src="../sources/LAF-LOG.png" width="45%" />
  <br />
  <img src="../sources/LAF-AF.png" width="45%" />
  <img src="../sources/LAF-AFR.png" width="45%" />
  <br />
  <img src="../sources/LAF-SQR.png" width="45%" />
  <img src="../sources/LAF-SQRT.png" width="45%" />
  <br />
</div>


This results in the following expected errors:

|    | Function   |   X Lower |   X Upper |   Mean Error |   Variance of Error |   Maximum Error |
|---:|:-----------|----------:|----------:|-------------:|--------------------:|----------------:|
|  0 | EXP        |   -8      |         8 |  -0.915841   |         4.59056     |      -0.0593698 |
|  1 | LOG        |    1      |        10 |   0.0538728  |         0.000650966 |       0.0340233 |
|  2 | AF         |   -1      |         1 |   0          |         0.0405663   |       0.376639  |
|  3 | AFR        |   -8      |         8 |  -0.00712797 |         0.000113342 |      -0.059711  |
|  4 | MUL        |   -1      |         1 |  -0.011893   |         0.000261394 |       0.110963  |
|  5 | SQR        |   -1      |         1 |   0.011893   |         0.000261394 |       0.110963  |
|  6 | SQRT       |    0.0001 |         1 |  -0.011894   |         0.000141425 |      -0.0604658 |

Most of the errors are quite low as found by the original paper. The graphs show that the LAF functions do roughly approximate the original functions quite closely.


### Software LAF C library: <laf_sw.h>
To solve the issue of implementing a neural network in C which uses normal RISC-V instructions to solve the XOR Problem, a new custom header file and its implementation had to be created. These are found at [laf_sw.h](../Vivado/RISCV_sources/sim/c_source/laf_sw.h) and [laf_sw.c](../Vivado/RISCV_sources/sim/c_source/laf_sw.c). 

These files define a custom math library as well as all the software implementations for the LAF functions, as the standard library is unavailable on freestanding environments, and many of the math functions that are usually convieniently available are not available. 

The implementation assumes 32 bit registers, hence only `float` and `int` types are used. This is important as some of the helper functions like `leftmostBit()` (which uses a technique called bit smearing for specifically 64 bit long) and the complex functions like `MUL(x, y)` and `SQR(val)` have to use such assumptions to prevent overflow errors.

### Software implementation of MLP: mlp_sw.c
Using the custom software LAF library, a whole program to perform MultiLayer Perceptron was written using C. This program works in freestanding environments (if not in `DEBUG` mode) which we need in the RISC V processing system. This file is found at [mlp_sw.c](../Vivado/RISCV_sources/sim/c_source/mlp_sw.c). 

In summary, the whole program works by assuming MSE Loss as a loss function and tanh as the activation function. By calculating the forward pass and then doing backpropagation and updating the weights and biases via Stochastic Gradient Descent, the model should be able to learn how to solve the XOR problem even with low counts of hidden neurons so long as there is enough epoches for it to reach the local minima for loss.

The backpropagation functions are as follows:

**Output Layer:**
   $$ \delta^{Output}_j = (a^{Output}_j - y_j) \cdot (1 - \tanh(z^{Output}_j)^2) $$
   $$ \frac{\partial E}{\partial w^{Hidden}_{ij}} = a^{Hidden}_i \cdot \delta^{Output}_j $$
   $$\frac{\partial E}{\partial b^{Output}_j} = \delta^{Output}_j$$ 

 **Hidden Layers:**
   $$\delta^{Layer}_i = \left(\sum_{j} w^{Layer}_{ij} \cdot \delta^{L+1}_j\right) \cdot (1 - \tanh(z^{Layer}_i)^2) $$
   $$\frac{\partial E}{\partial w^{Layer-1}_{ij}} = a^{Layer-1}_i \cdot \delta^{Layer}_j $$
   $$\frac{\partial E}{\partial b^{Layer}_i} = \delta^{Layer}_i $$

The update functions are as follows:

For weights:
$$w_{\text{new}} = w_{\text{old}} - \eta \cdot \nabla J(w_{\text{old}})$$

For biases:
$$b_{\text{new}} = b_{\text{old}} - \eta \cdot \nabla J(b_{\text{old}})$$ 

The choice for MSE loss is not very typical for a classification problem, since it better measures regression problem losses, but it happens to be the easiest to calculate with the least cycles for backpropagation.

Below is a screenshot of the model managing to solve the XOR problem with the hyperparameters of `HIDDEN_SIZE 2`, `LEARNING_RATE 0.01` and `NUM_EPOCHS 10000`. This was performed in a DEBUG environment outside of the RISC-V processor to test that the algorithm is functional.

![XOR Solved](../sources/MLP_C_Output.png)

Here is another non-linearly seperable problem for XNOR solved by the MLP. Hyperparameters are set to `HIDDEN_SIZE 5`, `LEARNING_RATE 0.01` and `NUM_EPOCHS 10000`.

![XNOR Solved](../sources/MLP_C_XNOR_Output.png)

> It is recommended to compile the software with RV32IM extensions for hardware multiplication and division. This is as having the compiler generate software multiplication wastes a lot of cycles and its not as far of a test. This also requires to edit constants.sv file in Vivado to enable the hardware MUL and DIV instructions.

> The lack of an RNG module severly hurts the convergence of the MLP to an optima. Initial weights cannot be 0, but having them all as the same small constant values causes symmetry issues.





### The need for randomness

Why random weights and gradients are needed?
From testing, an RNG module is required to ensure the convergence of the MLP to an optima as it is needed to generate initial weights and gradients are essential to ensuring that the model remains asymmetric, i.e the neurons compute different outputs during forward propagation and get different gradients during backpropagation. 

Making them all small constant values results in a model that never converges as shown below with the hyperparameters of `HIDDEN_SIZE 2`, `LEARNING_RATE 0.01` and `NUM_EPOCHS 10000`. All weights are initialized to 0.1 and weightGradients to 0.01.
![XOR Uncertain](../sources/MLP_C_XOR_ConstInit_Output.png)

The weights and biases for all the layers end up symmetric which leads to this bad prediction.

![XOR Symmetry](../sources/MLP_C_XOR_ConstInit_Weight_Bias.png)

## Future exploration and updates (Requires FPGA!)
### Making an RNG module - The quest for randomness

There are some ways to add a way to create random numbers in the RISC V processor, either by an RNG IP memory mapped to a certain address or making it a custom instruction which can load into CPU registers instead. The custom instruction method is cleaner to load random numbers into a temporary register. This approach mimicks Intel's `RDRAND` instruction in x64.

There are a few ways to implement pseudo-randomness quickly with [linear recurrences](https://en.wikipedia.org/wiki/Pseudorandom_number_generator#Generators_based_on_linear_recurrences),such as Linear Conguential Generator (LCG) and Linear Feedback Shift Registers (LFSR). LCG was chosen since its quite easy to implement, and has known [parameters for its moduli and multiplicator](https://statmath.wu.ac.at/software/src/prng-3.0.2/doc/prng.html/Table_LCG.html).

LCG generator formula: $X_{n+1} = (aX_n + c) \mod m$

For generating 32-bit integers, the following numbers are used (Park-Miller LCG):
- M (Modulus) = (1 << 31) - 1 = 2147483647 
- A (Multiplier) = 48271
- C (Increment) = 0
- Seed, $X_0$ = Any start value  

### Benchmarking instructions - RDTIME and RDCYCLE
As the RISC-V processor is a processing system (PS) seperate from the processing logic (PL) that is the ARM cores running PYNQ OS, there is no way to determine when it has completed its work...

To measure the clock cycles it takes to execute MLP model, RISC V defines two instructions that are of interest. In RISC-V, there are 3 CSR registers reserved for special counters, namely `cycle`, `time` and `instret`.

Of interest to us are the `cycle` and `time` counter, which can be accessed by user mode instruction `RDCYCLE` and `RDTIME`. 

`cycle` counter holds a count of number of clock cycles executed by the processor core on which the hart is running from an arbitrary start time in the past. 
`time` counter holds a count of wall-clock real time passed from an arbitrary start time in the past. 

In R32VI, to get the lower XLEN bits of the two counters, `RDCYCLE` and `RDTIME` are required, while the upper XLEN bits of the two counters are accessed with `RDCYCLEH` and `RDTIMEH`. The instructions follow the format as shown below:
![Time Counter Instructions](../sources/Time_Counters.png)

To note, these instructions are not necessarily implemented in the chip itself. Some cores such as [Sifive](https://forums.sifive.com/t/an-exception-occurred-while-using-rdtime/2915) implement their timers off chip and memory map it. Hardware is not required to do all timing. CSRs are optional, and some devices choose to trap to kernel to access off chip timers. So there are many approaches that can be taken! 

However, in the context of the problem of assuming a freestanding environment, let's just assume it has a requirement to be done on chip instead.

#### The on-chip way - Adding CSR regfiles

To add support for these instructions, we need a new regfile for CSR registers. At least the first 2 32-bit CSR registers (out of 64 possible 32-bit CSR registers) have to be working by the RISC-V specification to get `RDCYCLE` and `RDTIME`. The new regfile was implemented in [csr_regfile.sv](../Vivado/RISCV_sources/hdl/core/registers/csr_regfile.sv).

This also requires modification of the control path and the instruction decoder to pick up these two (or four if `RDCYCLEH` and `RDTIMEH` is also implemented) new instructions. 

With this instruction, we can make calls to the `RDCYCLE` instructions at the start of the program and the end of the program, and take the difference to determine how many cycles has actually occured. 


## Conclusions

While the multilayer perceptron has been written in C, it has been observed that it takes a lot more epoches compared to the Pytorch implementation to reach convergence likely due to the naive stochastic gradient descent algorithm I am using. The MLP model does manage to solve XOR problem with just 2 hidden neurons in the hidden layer.

An RNG module is required to ensure the convergence of the MLP to an optima. Random initial weights and gradients are essential to ensuring that the model remains asymmetric, i.e the neurons compute different outputs during forward propagation and get different gradients during backpropagation. Simply setting the weights and gradients initially to small constant values is insufficient to tackle this.

As such, more work has to be done on the RNG module and RDTIME modules. These two functions are essential to successfully benchmark the RISC-V CPU in machine learning tasks. RBF Neuron has not been implemented too.

Likewise, if an RNG module is created, a custom macro header file or toolchain has to be built to allow for RNG to work optimally.


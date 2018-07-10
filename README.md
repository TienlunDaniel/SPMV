# SPMV (Sparse Matrix-Vector Multiplication)
Many scientific data are represented and stored as matrices. However, sparse matrix is a special kind of matrix when there are many zero entries. Since zero times any number is still zero, if we can ignore all zero entries, we can receive a dramatic speed-up. Moreover, this SPMV program uses GPU's great computational power to do multiplications and additions since all the computations can be highly parallelized (CUDA programming).

# Requirement
Linux OS and Nvidia GPU(with CUDA toolkit is properly installed)

# Run the application
Makefile is provided. 

How to run: The code will need to take different number of command
line options defined as follows:<br/><br/>• -mat [matrixfile], input matrix file name. It’s a required parameter.
<br/>• -ivec [vectorfile], input vector file name. It’s required as well.
<br/>• -alg [approachOptions], this specifies which approach to be used for running
spmv code. It can be one of the following:<br/><br/>
-alg atom, use the simple atomics version
<br/>-alg segment, use the segment scan version
<br/>-alg design, use the designed version<br/>
<br/>• -blockSize [threadBlockSize], this specifies the thread block size to be used.
<br/>• -blockNum [threadBlockNum], this specifies the number of thread blocks to
be used.

# Report is in a separate file

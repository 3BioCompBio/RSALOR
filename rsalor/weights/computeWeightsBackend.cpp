#include "include/msa.h"

extern "C" float* computeWeightsBackend(
    const char* msa_path,
    unsigned int const msa_len,
    unsigned int const msa_depth,
    float seqid,
    bool count_target_sequence,
    unsigned int num_threads,
    bool verbose
)
{  

    // Init MSA
    MSA msa(
        msa_path, 
        msa_len,
        msa_depth,
        seqid,
        count_target_sequence,
        num_threads,
        verbose
    );

    // Check depth consistency
    unsigned int observed_msa_depth = msa.getDepth();
    if(observed_msa_depth != msa_depth) {
        std::cerr << "ERROR in computeWeights() (C++ backend): input msa_depth do not match to computed msa depth." << std::endl;
        std::cerr << " * msa_path:           " << msa_path << std::endl;
        std::cerr << " * input msa_depth:    " << msa_depth << std::endl;
        std::cerr << " * observed msa_depth: " << observed_msa_depth << std::endl;
        throw std::runtime_error("Invalid msa_depth argument");
    }

    // Allocate memory to the weights pointer because it will be passed to python
    float* weight_ptr = (float*)malloc(msa_depth*sizeof(float));
    auto weights_ptr_local = msa.getWeightsPointer();
    for(int i = 0; i < msa_depth; i++) { // Copy content from local
        weight_ptr[i]= weights_ptr_local[i];
    }
    return weight_ptr;

}

extern "C" void freeWeights(void* weights_ptr) {
    float* weights_ptr_casted = static_cast<float*>(weights_ptr);  
    if(weights_ptr_casted !=nullptr){
        delete [] weights_ptr_casted;
        weights_ptr_casted = nullptr;
    }
}
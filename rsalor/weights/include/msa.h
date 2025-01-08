#ifndef MSA_H
#define MSA_H

    #include<fstream>
    #include<iostream>
    #include<string>
    #include<vector>
    #include<unordered_map>
    #include<unordered_set>
    #include<numeric>
    #include<cstdint>  // For uint8_t type
    #include<thread>   // Standard C++ multi-threading
    //#include<chrono>   // To time code execution

    class MSA {
    protected:
        const char* msa_path;
        unsigned int msa_len;
        unsigned int msa_depth;
        float seqid;
        bool count_target_sequence;
        unsigned int num_threads;
        bool verbose;
        std::vector<std::vector<uint8_t>> seqs_int_form;
        std::vector<float> weights;

    public:
        
        // Constructor
        MSA(
            const char* msa_path,
            unsigned int msa_len,
            unsigned int msa_depth,
            float seqid,
            bool count_target_sequence,
            unsigned int num_threads,
            bool verbose
        );

        // Methods
        std::vector<std::vector<uint8_t>> readSequences();
        std::vector<float> computeWeights();
        void countClustersInRange(
            const std::vector<unsigned int>& range_indices,
            std::vector<unsigned int>& thread_counts,
            const unsigned int start_loop
        );

        // Getters
        float* getWeightsPointer();
        unsigned int getDepth();
        unsigned int getLength();
        float getNeff();
        
    };
#endif
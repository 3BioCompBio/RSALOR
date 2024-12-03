#ifndef MSA_H
#define MSA_H

    #include<fstream>
    #include<iostream>
    #include<string>
    #include<vector>
    #include<unordered_map>
    #include<unordered_set>
    #include<numeric>
    #include<cstdint>  // For uint8_t

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
        float* getWeightsPointer();
        unsigned int get_depth();
        unsigned int get_length();
        float get_Neff();

    };

#endif // MSA_H
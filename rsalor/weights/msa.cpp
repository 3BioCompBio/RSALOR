
// Header ----------------------------------------------------------------------
#if defined(_OPENMP)
    #include <omp.h>
#endif
#include "include/msa.h"


// -----------------------------------------------------------------------------

// MSA: Constructor ------------------------------------------------------------
MSA::MSA(
    const char* m_msa_path,
    unsigned int const m_msa_len,
    unsigned int const m_msa_depth,
    float m_seqid,
    bool m_count_target_sequence,
    unsigned int m_num_threads,
    bool m_verbose
):
msa_path(m_msa_path),
msa_len(m_msa_len),
msa_depth(m_msa_depth),
seqid(m_seqid),
count_target_sequence(m_count_target_sequence),
num_threads(m_num_threads),
verbose(m_verbose)
{
    // Read MSA
    if(this->verbose) {
        std::cout << "    - RSALOR (C++ backend): read sequences from file." << std::endl;
    }
    this->seqs_int_form = readSequences();

    // Compute weights
    if(this->verbose) {
        std::cout << "    - RSALOR (C++ backend): compute sequences weights." << std::endl;
    }
    this->weights = this->computeWeights();
}


// Methods ---------------------------------------------------------------------
std::vector<std::vector<uint8_t>> MSA::readSequences()
{

    // Init residues mapping to int
    std::unordered_map<char, uint8_t> res_mapping;
    res_mapping['A'] = 0;  res_mapping['C'] = 1;  res_mapping['D'] = 2;
    res_mapping['E'] = 3;  res_mapping['F'] = 4;  res_mapping['G'] = 5;
    res_mapping['H'] = 6;  res_mapping['I'] = 7;  res_mapping['K'] = 8;
    res_mapping['L'] = 9;  res_mapping['M'] = 10; res_mapping['N'] = 11;
    res_mapping['P'] = 12; res_mapping['Q'] = 13; res_mapping['R'] = 14;
    res_mapping['S'] = 15; res_mapping['T'] = 16; res_mapping['V'] = 17;
    res_mapping['W'] = 18; res_mapping['Y'] = 19; res_mapping['-'] = 20;
    res_mapping['.'] = 20; res_mapping['~'] = 20; res_mapping['B'] = 20;
    res_mapping['J'] = 20; res_mapping['O'] = 20; res_mapping['U'] = 20;
    res_mapping['X'] = 20; res_mapping['Z'] = 20;
    
    // Init
    std::vector<std::vector<uint8_t>> seqs_int_form;
    std::ifstream msa_file_stream(this->msa_path);
    std::string current_line;

    // Check file streaming
    if(msa_file_stream.fail()){
        std::cerr << "ERROR in MSA (C++ backend): Unable to open file." << this->msa_path << std::endl;
        throw std::runtime_error("Unable to open file containing the MSA data\n");
    }

    // Loop on lines of the file
    while(std::getline(msa_file_stream, current_line)){
        if(!current_line.empty() && current_line[0] != '>') { // Skip header and empty lines
            std::vector<uint8_t> current_seq_int;
            current_seq_int.reserve(this->msa_len); // optimize by putting the vector in the correct size which is known
            for (char c : current_line) {
                current_seq_int.push_back(res_mapping.at(toupper(c)));
            }
            seqs_int_form.push_back(current_seq_int);
        }
    }

    // Return
    return seqs_int_form;
}

// Compute sequences weight
std::vector<float> MSA::computeWeights()
{
    /*  Computes sequences weight.

        Returns
        -------
            weights   : The weigh of sequences computed with respect to 
                sequence identity obtained from this->seqid.

    */

    // Init
    std::vector<unsigned int> counts(this->msa_depth);
    for(unsigned int i = 0; i < this->msa_depth; ++i){
        counts[i] = 1;
    }

    // Remove first sequence from other weights computations by starting loop at 1
    unsigned int start_loop;
    if(this->count_target_sequence) {
        start_loop = 0; // count first sequence
    } else {
        start_loop = 1; // ignore first sequence
    }

    // Multi-thread case
    #if defined(_OPENMP)
        auto num_threads = this->num_threads;
        #pragma omp parallel num_threads(this->num_threads)
        {
            #pragma omp for schedule(dynamic)
            for (unsigned int i = start_loop; i < this->msa_depth; ++i) {
                auto& seq_i = this->seqs_int_form[i];
                unsigned int num_identical_residues;
                unsigned int num_aligned_residues;
                for (unsigned int j = start_loop; j < this->msa_depth; ++j) {
                    if (j >= i) continue; // Skip redundant comparisons or use (i != j) to exclude diagonal
                    auto& seq_j = this->seqs_int_form[j];
                    num_identical_residues = 0;
                    num_aligned_residues = 0;
                    for (unsigned int site = 0; site < this->msa_len; ++site) {
                        char char_i = seq_i[site];
                        char char_j = seq_j[site];
                        num_identical_residues += char_i == char_j;
                        num_aligned_residues += (char_i != 20 || char_j != 20);
                    }
                    if (static_cast<float>(num_identical_residues) > this->seqid * static_cast<float>(num_aligned_residues)) {
                        #pragma omp atomic
                        ++counts[i];
                        #pragma omp atomic
                        ++counts[j];
                    }
                }
            }
        }
    
    // Single-thread version
    #else
        for (unsigned int i = start_loop; i < this->msa_depth; ++i) {
            auto& seq_i = this->seqs_int_form[i];
            unsigned int num_identical_residues;
            unsigned int num_aligned_residues;
            for (unsigned int j = start_loop; j < this->msa_depth; ++j) {
                if (i >= j) continue; // Skip redundant comparisons or use (i != j) to exclude diagonal
                auto& seq_j = this->seqs_int_form[j];
                num_identical_residues = 0;
                num_aligned_residues = 0;
                for (unsigned int site = 0; site < this->msa_len; ++site) {
                    char char_i = seq_i[site];
                    char char_j = seq_j[site];
                    num_identical_residues += char_i == char_j;
                    num_aligned_residues += (char_i != 20 || char_j != 20);
                }
                if (static_cast<float>(num_identical_residues) > this->seqid * static_cast<float>(num_aligned_residues)) {
                    ++counts[i];
                    ++counts[j];                    
                }
            }
        }
    #endif
    
    // Convert counts to weights
    std::vector<float> weights(this->msa_depth);
    for(unsigned int i = 0; i < this->msa_depth; ++i){
        weights[i] = 1.f/ static_cast<float>(counts[i]);
    }

    // Remove first sequences weight (that was initally assigned to 1.0)
    if(!this->count_target_sequence) {
        weights[0] = 0.f;
    }

    // Return
    return weights;
}

// Function to return a pointer to the weights
float* MSA::getWeightsPointer() {
    return weights.data();
}

// Getters
unsigned int MSA::get_depth() {
    return this->msa_depth;
}

unsigned int MSA::get_length() {
    return this->msa_len;
}

float MSA::get_Neff() {
    return std::accumulate(this->weights.begin(), this->weights.end(), 0.f);
}
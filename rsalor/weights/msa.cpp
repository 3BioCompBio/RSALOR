
// Header ----------------------------------------------------------------------
#include "include/msa.h"

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

// Parse MSA sequences from file  ----------------------------------------------
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

// Assign weights for all sequences based on clusters --------------------------

// Compute sequences weight
std::vector<float> MSA::computeWeights(){

    // Init counts (all threads)
    std::vector<unsigned int> counts(this->msa_depth, 1);

    // Count or ignore first sequence for weights computations by starting loop at 0 or 1
    unsigned int start_loop = this->count_target_sequence ? 0 : 1;

    // Initialize the per-thread counts vectors
    std::vector<std::vector<unsigned int>> thread_counts(
        num_threads, std::vector<unsigned int>(this->msa_depth, 0)
    );

    // Separate indices in chunks for each thread
    // * Trick: Since we only loop on half (i, j)-matrix (j < i), first i iterations will stop much earlier than last,
    //          so we distribute i indices evenly across threads, so they all terminate approximatively at the same time
    std::vector<std::vector<unsigned int>> threads_indices(num_threads);
    for (unsigned int i = start_loop; i < this->msa_depth; ++i) {
        unsigned int thread_id = i % num_threads;
        threads_indices[thread_id].push_back(i);
    }

    // Manage multi-threading
    std::vector<std::thread> threads;
    for (unsigned int t = 0; t < num_threads; ++t) {
        threads.emplace_back( // ok here some magic
            [this, &threads_indices, &thread_counts, t, start_loop]() {
            countClustersInRange(threads_indices[t], thread_counts[t], start_loop); // compute cluster by chunks
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    // Merge thread counts into global counts
    for (const auto& thread_count : thread_counts) {
        for (unsigned int i = 0; i < this->msa_depth; ++i) {
            counts[i] += thread_count[i];
        }
    }

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

void MSA::countClustersInRange(
    const std::vector<unsigned int>& range_indices,
    std::vector<unsigned int>& range_counts,
    const unsigned int start_loop
)
{
    // Init
    unsigned int num_identical_residues;
    unsigned int identical_residues_thr = static_cast<unsigned int>(this->seqid * this->msa_len);

    // Loop on range
    for (auto i : range_indices) {
        const auto& seq_i = this->seqs_int_form[i];
        // Loop on other sequences j < i (half matrix because (i, i)=(j, i))
        for (unsigned int j = start_loop; j < i; ++j) {
            const auto& seq_j = this->seqs_int_form[j];
            
            // Compute seqid(i, j)
            num_identical_residues = 0;
            for (unsigned int site = 0; site < this->msa_len; ++site) {
                num_identical_residues += seq_i[site] == seq_j[site];
            }
            
            // Update if (i, j) in same cluster
            if (num_identical_residues > identical_residues_thr) {
                ++range_counts[i];
                ++range_counts[j];
            }
        }
    }
}

// Getter ----------------------------------------------------------------------
float* MSA::getWeightsPointer() {
    return weights.data();
}

// Getters
unsigned int MSA::getDepth() {
    return this->msa_depth;
}

unsigned int MSA::getLength() {
    return this->msa_len;
}

float MSA::getNeff() {
    return std::accumulate(this->weights.begin(), this->weights.end(), 0.f);
}
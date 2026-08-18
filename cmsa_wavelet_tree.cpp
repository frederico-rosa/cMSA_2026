// =============================================================
// cmsa_wavelet_tree.cpp
//
// cMSA usando wavelet tree da SDSL como estrutura comprimida.
//
// Port do case 5 (cMSA_sdsl, wavelet tree) de msa_v07_wt.cpp para o
// mesmo estilo dos demais programas msa_v0X: leitura de
// bigann_learn.bvecs_100M.txt no formato [objetos][referencias]
// (em vez do formato [referencias][objetos] que msa_v07_wt.cpp usa),
// argumentos de linha de comando posicionais (objs refs ques), e os
// mesmos prints de tempo/memoria/resultado dos outros arquivos.
//
// A LOGICA do algoritmo NAO foi alterada: createOrderedList/
// createPositionList, a escrita do array L por objeto, a construcao
// da wavelet tree via sdsl::construct() e a busca em
// fullPermutationSearching_sdsl (aqui renomeada searchByObject, com
// os parametros acc/M removidos por serem sempre NULL nesse case)
// sao copias fieis do msa_v07_wt.cpp original — so' os indices que
// podem exceder 32 bits (produtos com nn) foram colocados em
// long long, seguindo o mesmo tratamento ja aplicado aos outros
// arquivos desta serie (msa_v01_long.cpp, msa_v01_otimizada.cpp
// etc.); os VALORES armazenados na wavelet tree continuam sempre
// < refs, nunca precisando de mais que 32 bits.
//
// Todo o restante de msa_v07_wt.cpp (algoritmos A1-A4, A7, A9,
// Simple-9, delta, busca exata, getopt, escrita em arquivo -o) foi
// removido por nao ser usado pelo case 5.
// =============================================================
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include <sdsl/int_vector.hpp>
#include <sdsl/wavelet_trees.hpp>

using namespace std;
using namespace sdsl;

typedef wt_gmr<> wt_sdsl; // mesma escolha de msa_v07_wt.cpp (-> OK)

// =========================================================
// CONFIGURAÇÕES
// =========================================================
// arquivo temporario exigido pela API do sdsl::construct() (le a
// sequencia de um arquivo em disco); mesmo nome usado no original.
static const std::string STRUCTURE_TMP_FILE = "tmp.sdsl";
static const int KNN = 10;

// =========================================================
// CONTEXT
// =========================================================
struct Context {
    int nn = 0;
    int n = 0;
    int dim = 0;
    int num_q = 0;
};

struct Node {
    int i;
    double d;
};

struct CompareNode {
    bool operator()(Node const& a, Node const& b) const {
        if (a.d != b.d) return a.d < b.d;
        return a.i < b.i;
    }
};

typedef priority_queue<Node, vector<Node>, CompareNode> priority_queue_Node;

// =========================================================
// MEMÓRIA
// =========================================================
static long getVmRSSKB() {

    ifstream in("/proc/self/status");
    string key;

    while (in >> key) {
        if (key == "VmRSS:") {
            long value;
            string unit;
            in >> value >> unit;
            return value;
        }
        string line;
        getline(in, line);
    }

    return -1;
}

static void printMemory(const string& msg) {
    long rssKB = getVmRSSKB();
    cout << msg << " : " << rssKB / 1024.0 << " MB" << endl;
}

// =========================================================
// ORDENAÇÃO / LISTA DE POSIÇÕES
// (identico a node_sorter/createOrderedList/createPositionList de
// msa_v07_wt.cpp)
// =========================================================
static bool node_sorter(const Node& lhs, const Node& rhs) {
    if (lhs.d != rhs.d) return lhs.d < rhs.d;
    return lhs.i < rhs.i;
}

static void createOrderedList(int refs, int dim, const double* oi, int* loi, const double* R) {

    vector<Node> output(refs);

    for (int i = 0; i < refs; i++) {
        double d = 0.0;
        for (int kk = 0; kk < dim; kk++) {
            double diff = oi[kk] - R[dim * i + kk];
            d += diff * diff;
        }
        output[i] = Node{i, d};
    }

    std::sort(output.begin(), output.end(), node_sorter);

    for (int i = 0; i < refs; i++) {
        loi[i] = output[i].i;
    }
}

static void createPositionList(int refs, int dim, const double* oi, int* lpi, const double* R) {

    vector<Node> output(refs);

    for (int i = 0; i < refs; i++) {
        double d = 0.0;
        for (int kk = 0; kk < dim; kk++) {
            double diff = oi[kk] - R[dim * i + kk];
            d += diff * diff;
        }
        output[i] = Node{i, d};
    }

    std::sort(output.begin(), output.end(), node_sorter);

    for (int i = 0; i < refs; i++) {
        lpi[output[i].i] = i;
    }
}

// =========================================================
// FILA DE PRIORIDADE DE TAMANHO FIXO (identico a
// init_pq_fixed_size/push_pq_fixed_size de msa_v07_wt.cpp)
// =========================================================
static void initPQFixedSize(priority_queue_Node& PQ, int knn) {
    Node tmp{0, DBL_MAX};
    for (int i = 0; i < knn; i++) {
        PQ.push(tmp);
    }
}

static void pushPQFixedSize(priority_queue_Node& PQ, Node tmp) {
    if (tmp.d < PQ.top().d) {
        PQ.pop();
        PQ.push(tmp);
    }
}

// =========================================================
// BUSCA
// (identico a fullPermutationSearching_sdsl de msa_v07_wt.cpp; os
// parametros "acc" e "M", sempre NULL/nao usados nesse case, foram
// removidos)
// =========================================================
static void searchByObject(int refs, int dim, long long n, const double* q, const double* R,
                            wt_sdsl& WT, priority_queue_Node& PQ, int knn) {

    vector<int> lq(refs);
    createOrderedList(refs, dim, q, lq.data(), R);

    vector<int> inv_lq(refs);
    for (int i = 0; i < refs; i++) {
        inv_lq[lq[i]] = i;
    }

    double acc = 0.0;
    long long total = n * static_cast<long long>(refs);

    for (long long j = 0; j < total; j++) {

        uint32_t refPos = WT[j];
        int idxRefPos = static_cast<int>(j % refs);
        int refPosQ = inv_lq[refPos];

        acc += std::abs(refPosQ - idxRefPos);

        if (idxRefPos == refs - 1) {
            long long oid = j / refs;
            Node tmp{static_cast<int>(oid), acc};
            pushPQFixedSize(PQ, tmp);
            acc = 0.0;
        }
    }
}

// =========================================================
// impressão do resultado (mesma logica de write_output de
// msa_v07_wt.cpp: inverte a fila via pilha para sair em ordem
// ascendente por score)
// =========================================================
static void printResult(priority_queue_Node& PQ) {
    vector<Node> stack;
    while (!PQ.empty()) {
        stack.push_back(PQ.top());
        PQ.pop();
    }
    for (auto it = stack.rbegin(); it != stack.rend(); ++it) {
        cout << it->i << " ";
    }
    cout << endl;
}

// =========================================================
// MAIN
// =========================================================
int main(int argc, char* argv[]) {

    if (argc < 5) {
        cerr << "Uso: " << argv[0] << " <SEL_OBJS> <SEL_REFS> <SEL_QUES> <arquivo_entrada>" << endl;
        return 1;
    }

    int SEL_OBJS = std::stoi(argv[1]);
    int SEL_REFS = std::stoi(argv[2]);
    int SEL_QUES = std::stoi(argv[3]);
    std::string FILE_NAME = argv[4];

    cout << "cmsa_wavelet_tree.cpp" << endl;
    cout << "Programa: cMSA com wavelet tree (SDSL), a partir do case 5 de msa_v07_wt.cpp" << endl;
    printMemory("Antes de comecar o programa");

    Context ctx;

    ifstream in(FILE_NAME);
    if (!in) {
        cerr << "Erro abrindo arquivo" << endl;
        return 1;
    }

    in >> ctx.dim >> ctx.nn >> ctx.n >> ctx.num_q;

    ctx.nn = SEL_OBJS;
    ctx.n = SEL_REFS;
    ctx.num_q = SEL_QUES;

    cout << "n = " << ctx.n << endl;
    cout << "nn = " << ctx.nn << endl;
    cout << "dim = " << ctx.dim << endl;
    cout << "num_q = " << ctx.num_q << endl;

    long long msaBytes = 4LL * ctx.n * ctx.nn;
    printf("MSA teorico = %.2f MB\n", msaBytes / 1024.0 / 1024.0);

    vector<double> r(static_cast<size_t>(ctx.dim) * ctx.n);

    {
        double tmp;
        long long total = static_cast<long long>(ctx.dim) * ctx.nn;
        for (long long i = 0; i < total; i++) {
            in >> tmp;
        }
    }

    for (size_t i = 0; i < r.size(); i++) {
        in >> r[i];
    }

    printMemory("Apos referencias");

    in.close();

    /* ***************************************
       ************ CONSTRUCT ****************
       *************************************** */
    auto t1 = chrono::steady_clock::now();

    ifstream in2(FILE_NAME);
    {
        int tmp;
        for (int i = 0; i < 4; i++) {
            in2 >> tmp;
        }
    }

    ofstream outfile(STRUCTURE_TMP_FILE, ios::out | ios::binary);
    if (!outfile) {
        cerr << "Erro ao criar " << STRUCTURE_TMP_FILE << endl;
        return 1;
    }

    vector<double> oi(ctx.dim);
    vector<int> lpi(ctx.n);
    vector<uint32_t> L(static_cast<size_t>(ctx.n));

    long long totalDimNn = static_cast<long long>(ctx.dim) * ctx.nn;

    for (long long i = 0; i < totalDimNn; i++) {

        in2 >> oi[i % ctx.dim];
        if (i % ctx.dim == ctx.dim - 1) {

            createPositionList(ctx.n, ctx.dim, oi.data(), lpi.data(), r.data());

            for (int j = 0; j < ctx.n; j++) {
                L[lpi[j]] = static_cast<uint32_t>(j);
            }

            for (int k = 0; k < ctx.n; k++) {
                outfile.write(reinterpret_cast<const char*>(&L[k]), sizeof(uint32_t));
            }
        }
    }

    outfile.close();

    wt_sdsl WT;
    construct(WT, STRUCTURE_TMP_FILE, sizeof(uint32_t));

    auto t2 = chrono::steady_clock::now();
    printMemory("Apos construcao");
    printf("cMSA size = %.2f MB\n", size_in_bytes(WT) / 1024.0 / 1024.0);
    printf("TEMPO CONSTRUCAO = %lld ms\n",
           static_cast<long long>(chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()));

    // pula novamente o bloco de referencias, para chegar as queries
    // (mesmo padrao de "re-consumo" ja usado nos demais msa_v0X, pois
    // in2 comecou do zero e ja recomeu o bloco de objetos)
    {
        double tmp;
        for (size_t i = 0; i < r.size(); i++) {
            in2 >> tmp;
        }
    }

    /* ***************************************
       ************** SEARCH  ****************
       *************************************** */
    for (int qid = 0; qid < ctx.num_q; qid++) {

        vector<double> q(ctx.dim);
        for (int j = 0; j < ctx.dim; j++) {
            in2 >> q[j];
        }

        priority_queue_Node PQ;
        initPQFixedSize(PQ, KNN);

        searchByObject(ctx.n, ctx.dim, ctx.nn, q.data(), r.data(), WT, PQ, KNN);

        printResult(PQ);
    }

    in2.close();

    auto t3 = chrono::steady_clock::now();
    printMemory("Depois da busca");
    printf("TEMPO BUSCA = %lld ms\n",
           static_cast<long long>(chrono::duration_cast<chrono::milliseconds>(t3 - t2).count()));

    return 0;
}

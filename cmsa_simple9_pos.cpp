// =============================================================
// cmsa_simple9_pos.cpp
//
// Simple-9 com posicao direta (sem estado de diferenca).
//
// Traducao para C++ do programa Java msa_v05_cmsa_simple9.
// =============================================================
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <stdexcept>

#if defined(__linux__)
#include <fstream>
#endif

// =========================================================
// CONFIGURAÇÕES
// =========================================================
static const int KNN = 10; // k-NN // número de objetos mais próximos de uma query

// =========================================================
// CONTEXT
// =========================================================
struct Context {
    int nn = 0;
    int n = 0;
    int dim = 0;
    int num_q = 0;

    std::vector<int> offset;
};

struct Node {
    int i;
    double d;

    Node() : i(0), d(0.0) {}
    Node(int i_, double d_) : i(i_), d(d_) {}
};

// =========================================================
// Simple9Encoder
// =========================================================
class Simple9Encoder {
public:
    // quantidade de inteiros e bits de cada modo
    static const int LIMITS[9];
    static const int BITS[9];

    // buffer temporário
    int buffer[28];
    int bufferSize = 0;

    // modo de operação
    bool countOnly;

    // saída
    std::vector<int> output; // usado apenas na segunda passagem
    int outPos = 0;

    // contador
    int wordCount = 0;

    // ---------------------------------------------------
    // PRIMEIRA PASSAGEM
    // ---------------------------------------------------
    Simple9Encoder() : countOnly(true) {}

    // ---------------------------------------------------
    // SEGUNDA PASSAGEM
    // ---------------------------------------------------
    explicit Simple9Encoder(int outputSize) : countOnly(false) {
        output.assign(outputSize, 0);
    }

    // ---------------------------------------------------
    // adiciona inteiro
    // ---------------------------------------------------
    void add(int value) {
        buffer[bufferSize++] = value;

        if (bufferSize == 28) {
            flush();
        }
    }

    // ---------------------------------------------------
    // finalizar
    // ---------------------------------------------------
    void finish() {
        while (bufferSize > 0) {
            flush();
        }
    }

    // ---------------------------------------------------
    // empacotamento
    // ---------------------------------------------------
    void flush() {
        for (int sel = 0; sel < 9; sel++) {

            int count = std::min(LIMITS[sel], bufferSize);
            int bits = BITS[sel];

            bool ok = true;

            for (int i = 0; i < count; i++) {
                if (buffer[i] >= (1 << bits)) {
                    ok = false;
                    break;
                }
            }

            if (!ok) {
                continue;
            }

            // -------------------------------
            // monta a word
            // -------------------------------
            int word = sel << 28;

            for (int i = 0; i < count; i++) {
                word |= buffer[i] << (28 - bits * (i + 1));
            }

            // -------------------------------
            // conta ou grava
            // -------------------------------
            if (countOnly) {
                wordCount++;
            } else {
                output[outPos++] = word;
            }

            // -------------------------------
            // remove elementos utilizados
            // -------------------------------
            int remaining = bufferSize - count;

            for (int i = 0; i < remaining; i++) {
                buffer[i] = buffer[i + count];
            }

            bufferSize = remaining;

            return;
        }

        throw std::runtime_error("Valor incompativel com Simple-9.");
    }
};

const int Simple9Encoder::LIMITS[9] = {28, 14, 9, 7, 5, 4, 3, 2, 1};
const int Simple9Encoder::BITS[9]   = {1, 2, 3, 4, 5, 7, 9, 14, 28};

// =========================================================
// Simple9Decoder
// =========================================================
class Simple9Decoder {
public:
    static const int LIMITS[9];
    static const int BITS[9];

    const std::vector<int>* input;

    int wordPos = 0;

    int buffer[28];
    int bufferSize = 0;
    int bufferPos = 0;

    explicit Simple9Decoder(const std::vector<int>& input_) : input(&input_) {}

    int next() {
        if (bufferPos >= bufferSize) {

            if (wordPos >= static_cast<int>(input->size())) {
                return -1;
            }

            uint32_t word = static_cast<uint32_t>((*input)[wordPos++]);

            int sel = static_cast<int>(word >> 28);

            int count = LIMITS[sel];
            int bits = BITS[sel];

            uint32_t mask = (1u << bits) - 1u;

            bufferSize = count;
            bufferPos = 0;

            for (int i = 0; i < count; i++) {
                int shift = 28 - bits * (i + 1);
                buffer[i] = static_cast<int>((word >> shift) & mask);
            }
        }

        return buffer[bufferPos++];
    }

    void reset() {
        wordPos = 0;
        bufferPos = 0;
        bufferSize = 0;
    }
};

const int Simple9Decoder::LIMITS[9] = {28, 14, 9, 7, 5, 4, 3, 2, 1};
const int Simple9Decoder::BITS[9]   = {1, 2, 3, 4, 5, 7, 9, 14, 28};

// =========================================================
// Funções auxiliares
// =========================================================
static void printMemory(const std::string& msg) {
    double usedMB = 0.0;

#if defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string line;
    long vmRssKb = -1;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::istringstream iss(line.substr(6));
            iss >> vmRssKb;
            break;
        }
    }
    if (vmRssKb >= 0) {
        usedMB = vmRssKb / 1024.0;
    }
#endif

    std::printf("%s : %.2f MB\n", msg.c_str(), usedMB);
}

static std::vector<int> getOrderedList(const Context& ctx, const std::vector<double>& oi, const std::vector<double>& r) {

    std::vector<Node> loi(ctx.n); // ordered list

    for (int k = 0; k < ctx.n; k++) {

        double term = 0;
        double dist = 0;

        for (int j = 0; j < ctx.dim; j++) {
            term = oi[j] - r[k * ctx.dim + j];
            dist += term * term;
        }

        loi[k] = Node(k, dist);
    }

    std::sort(loi.begin(), loi.end(), [](const Node& a, const Node& b) {
        if (a.d != b.d) {
            return a.d < b.d;
        }
        return a.i < b.i;
    });

    std::vector<int> output(ctx.n);

    for (int k = 0; k < ctx.n; k++) {
        output[k] = loi[k].i;
    }

    return output;
}

// Calcula, em O(n) (uma unica passada sobre a lista ja ordenada), o
// mapeamento inverso pos[ref] = posicao(rank). Substitui n chamadas a
// getPosition() (cada uma O(n)) por n atribuicoes O(1), eliminando o
// custo O(n^2) por objeto que dominava a construcao para n grande.
static std::vector<int> invertPositions(const Context& ctx, const std::vector<int>& loi) {
    std::vector<int> pos(ctx.n);
    for (int rank = 0; rank < ctx.n; rank++) {
        pos[loi[rank]] = rank;
    }
    return pos;
}

static void print(const std::vector<int>& a) {
    std::ostringstream out;
    for (size_t i = 0; i < a.size(); i++) {
        out << a[i] << " ";
    }
    std::cout << out.str() << std::endl;
}

// constroi o codigo Simple-9 codificando a POSICAO diretamente (nao a
// diferenca entre objetos consecutivos no mesmo bucket)
static void constructSimple9(const std::vector<int>& pos, const Context& ctx,
                              std::vector<Simple9Encoder>& buckets) {

    for (int k = 0; k < ctx.n; k++) {

        // Antes: codificava a DIFERENCA entre o valor absoluto deste
        // objeto (obj*ctx.n+pos[k]) e o do objeto anterior no mesmo
        // bucket — exigia rastrear firstElement[k]/lastValue[k] entre
        // objetos, e a diferenca podia chegar a quase 2*ctx.n no pior
        // caso (quando a posicao "pula" de 0 para ctx.n-1, ou vice-
        // versa, entre dois objetos consecutivos).
        //
        // Agora: codifica a posicao diretamente (0..ctx.n-1). Nao ha'
        // bias necessario (diferente do codigo gamma/delta do v03/v04,
        // o Simple-9 representa 0 sem problema), nem estado entre
        // objetos — cada posicao e' autocontida. Valor maximo cai de
        // ~2n para n.
        buckets[k].add(pos[k]);
    }
}

// obtem a distância um objeto por vez, diferentemente da versão original
// que percorre um bucket por vez e constroi as distâncias dos objetos
// usando um array de tamanho nn (impacta RAM)
static void searchByObject(const std::vector<int>& lq, std::vector<Simple9Decoder>& dec,
                            const Context& ctx) {

    // MAX-HEAP: pior elemento no topo (maior d; empate -> menor i no topo)
    auto cmp = [](const Node& a, const Node& b) {
        if (a.d != b.d) {
            return a.d < b.d;
        }
        return a.i > b.i;
    };
    std::priority_queue<Node, std::vector<Node>, decltype(cmp)> heap(cmp);

    for (int i = 0; i < ctx.nn; i++) {

        int score = 0;

        // busca nos buckets referentes a este objeto
        for (int k = 0; k < ctx.n; k++) {

            int bucketID = lq[k];

            // Cada posicao e' autocontida agora: basta decodificar,
            // sem acumular estado entre objetos.
            int refPos = dec[bucketID].next();

            score += std::abs(refPos - k);
        }

        // =====================================
        // atualiza TOP-K
        // =====================================
        Node node(i, score);

        if (static_cast<int>(heap.size()) < KNN) {

            heap.push(node);

        } else {

            const Node& worst = heap.top();

            if (score < worst.d) {
                heap.pop();
                heap.push(node);
            }
        }
    }

    // =====================================
    // converte heap -> array ordenado
    // =====================================
    std::vector<Node> top;
    top.reserve(heap.size());

    while (!heap.empty()) {
        top.push_back(heap.top());
        heap.pop();
    }

    // heap sai invertida
    std::sort(top.begin(), top.end(), [](const Node& a, const Node& b) {
        if (a.d != b.d) {
            return a.d < b.d;
        }
        return a.i < b.i;
    });

    std::vector<int> result(top.size());

    for (size_t i = 0; i < top.size(); i++) {
        result[i] = top[i].i;
    }

    print(result);
}

// =========================================================
// MAIN
// =========================================================
int main(int argc, char* argv[]) {

    if (argc < 5) {
        std::cerr << "Uso: " << argv[0] << " <SEL_OBJS> <SEL_REFS> <SEL_QUES> <arquivo_entrada>" << std::endl;
        return 1;
    }

    int SEL_OBJS = std::stoi(argv[1]);
    int SEL_REFS = std::stoi(argv[2]);
    int SEL_QUES = std::stoi(argv[3]);
    std::string FILE_NAME = argv[4];

    std::cout << "cmsa_simple9_pos.cpp" << std::endl;
	std::cout << "Programa: cMSA otimizada simple9 direto em RAM (sem armazenar em disco)" << std::endl;
    printMemory("Antes de comecar o programa");

    Context ctx;

    // First read: storing references in RAM
    std::ifstream sc(FILE_NAME);
    if (!sc.is_open()) {
        std::cerr << "Nao foi possivel abrir o arquivo: " << FILE_NAME << std::endl;
        return 1;
    }

    sc >> ctx.dim;
    sc >> ctx.nn;
    sc >> ctx.n;
    sc >> ctx.num_q;

    ctx.nn = SEL_OBJS;
    ctx.n = SEL_REFS;
    ctx.num_q = SEL_QUES;

    std::cout << "n = " << ctx.n << std::endl;
    std::cout << "nn = " << ctx.nn << std::endl;
    std::cout << "dim = " << ctx.dim << std::endl;
    std::cout << "num_q = " << ctx.num_q << std::endl;

    long long msaBytes = 4LL * ctx.n * ctx.nn;

    std::printf("MSA teorico = %.2f MB\n", msaBytes / 1024.0 / 1024.0);

    std::vector<double> r(static_cast<size_t>(ctx.dim) * ctx.n);

    {
        double tmp;
        long long total = static_cast<long long>(ctx.dim) * ctx.nn;
        for (long long i = 0; i < total; i++) {
            sc >> tmp;
        }
    }

    for (size_t i = 0; i < r.size(); i++) {
        sc >> r[i];
    }

    printMemory("Apos referencias");

    sc.close();

    /* ***************************************
       ************ CONSTRUCT ****************
       *************************************** */
    // Read objects and construct MSA
    auto t1 = std::chrono::steady_clock::now();

    // Second read: calculate memory space required to cMSA
    sc.open(FILE_NAME);
    {
        int tmp;
        for (int i = 0; i < 4; i++) {
            sc >> tmp;
        }
    }

    std::vector<double> oi(ctx.dim, 0);

    // contabiliza o número de bytes para construção de estrutura
    std::vector<Simple9Encoder> counters(ctx.n);

    long long totalDimNn = static_cast<long long>(ctx.dim) * ctx.nn;

    for (long long i = 0; i < totalDimNn; i++) {

        sc >> oi[i % ctx.dim];
        if (i % ctx.dim == ctx.dim - 1) {
            std::vector<int> loi = getOrderedList(ctx, oi, r);
            std::vector<int> pos = invertPositions(ctx, loi);
            constructSimple9(pos, ctx, counters);
        }
    }

    for (int k = 0; k < ctx.n; k++) {
        counters[k].finish();
    }

    sc.close();

    // total de bytes alocados
    double cMsaSize = 0.0;
    for (const Simple9Encoder& c : counters) {
        cMsaSize += c.wordCount;
    }

    std::printf("cMSA size = %1.2f MB\n", cMsaSize * 4 / 1024 / 1024);

    // Third read: populate cMSA memory space
    sc.open(FILE_NAME);
    {
        int tmp;
        for (int i = 0; i < 4; i++) {
            sc >> tmp;
        }
    }

    std::vector<Simple9Encoder> bucketsS9;
    bucketsS9.reserve(ctx.n);
    for (int k = 0; k < ctx.n; k++) {
        bucketsS9.emplace_back(counters[k].wordCount);
    }

    for (long long i = 0; i < totalDimNn; i++) {

        sc >> oi[i % ctx.dim];
        if (i % ctx.dim == ctx.dim - 1) {
            std::vector<int> loi = getOrderedList(ctx, oi, r);
            std::vector<int> pos = invertPositions(ctx, loi);
            constructSimple9(pos, ctx, bucketsS9);
        }
    }

    for (int k = 0; k < ctx.n; k++) {
        bucketsS9[k].finish();
    }

    auto t2 = std::chrono::steady_clock::now();
    printMemory("Apos construcao");
    std::printf("TEMPO CONSTRUCAO = %lld ms\n",
                static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()));

    // MSA construction finished
    {
        double tmp;
        for (size_t i = 0; i < r.size(); i++) {
            sc >> tmp;
        }
    }

    /* ***************************************
       ************** SEARCH  ****************
       *************************************** */
    // Queries start
    std::vector<Simple9Decoder> dec;
    dec.reserve(ctx.n);
    for (int k = 0; k < ctx.n; k++) {
        dec.emplace_back(bucketsS9[k].output);
    }

    for (int qid = 0; qid < ctx.num_q; qid++) {

        std::vector<double> q(ctx.dim);

        for (int j = 0; j < ctx.dim; j++) {
            sc >> q[j];
        }

        std::vector<int> lq = getOrderedList(ctx, q, r);

        searchByObject(lq, dec, ctx);

        for (int j = 0; j < ctx.n; j++) {
            dec[j].reset();
        }
    }

    sc.close();

    auto t3 = std::chrono::steady_clock::now();
    printMemory("Depois da busca");
    std::printf("TEMPO BUSCA = %lld ms\n",
                static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()));

    return 0;
}
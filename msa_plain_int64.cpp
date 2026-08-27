// =============================================================
// msa_plain_int64.cpp
//
// Igual a msa_plain.cpp, mas com indices/array em long long para
// nao estourar 32 bits em datasets grandes.
// =============================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

using namespace std;

// =========================================================
// CONFIGURAÇÕES
// =========================================================
static const int KNN = 10;

// =========================================================
// CONTEXT
// =========================================================
struct Context {
    int nn;
    int n;
    int dim;
    int num_q;
};

struct Node {
    int i;
    double d;

    Node(int ii, double dd) : i(ii), d(dd) {}
};

struct CompareMaxHeap {
    bool operator()(const Node& a, const Node& b) const {

        if (a.d != b.d)
            return a.d < b.d;

        return a.i < b.i;
    }
};

// =========================================================
// MEMÓRIA
// =========================================================

long getVmRSSKB() {

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

void printMemory(const string& msg) {

    long rssKB = getVmRSSKB();

    cout << msg << " : " << rssKB / 1024.0 << " MB" << endl;
}

// =========================================================

void printVector(const vector<int>& a) {

    for (int v : a)
        cout << v << " ";

    cout << endl;
}

int getPosition(int ref, const vector<int>& loi) {

    for (int i = 0; i < (int)loi.size(); i++) {

        if (ref == loi[i])
            return i;
    }

    return -1;
}

// =========================================================

// Pula "count" tokens (separados por espaco/quebra de linha) do stream sem
// converter cada um para double. O operator>> usado no resto do arquivo
// faz parsing numerico a cada leitura, o que fica proibitivamente lento
// quando ha bilhoes de tokens a descartar (ex.: selecionar poucos objetos
// de um arquivo com dezenas de milhoes deles).
void skipTokens(ifstream& in, long long count) {

    if (count <= 0)
        return;

    const size_t BUF_SIZE = 1 << 20;
    vector<char> buf(BUF_SIZE);

    long long skipped = 0;
    bool inToken = false;

    while (skipped < count && in) {

        in.read(buf.data(), BUF_SIZE);
        streamsize got = in.gcount();

        if (got == 0)
            break;

        for (streamsize k = 0; k < got; k++) {

            char c = buf[k];

            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {

                if (inToken) {

                    inToken = false;
                    skipped++;

                    if (skipped == count) {

                        in.clear();
                        in.seekg(-(got - k - 1), ios::cur);
                        return;
                    }
                }

            } else {

                inToken = true;
            }
        }
    }
}

// =========================================================

vector<int> getOrderedList(const Context& ctx, const vector<double>& oi, const vector<double>& r) {

    vector<Node> loi;
    loi.reserve(ctx.n);

    for (int k = 0; k < ctx.n; k++) {

        double dist = 0.0;

        for (int j = 0; j < ctx.dim; j++) {

            double term = oi[j] - r[k * ctx.dim + j];

            dist += term * term;
        }

        loi.emplace_back(k, dist);
    }

    sort(loi.begin(), loi.end(), [](const Node& a, const Node& b) {

             if (a.d != b.d)
                 return a.d < b.d;

             return a.i < b.i;
         });

    vector<int> output(ctx.n);

    for (int k = 0; k < ctx.n; k++)
        output[k] = loi[k].i;

    return output;
}

// =========================================================
// n x nn pode superar 2^31-1 (limite de um int de 32 bits) quando os
// testes usam entradas grandes. msa passa a ser vector<long long> e o
// indice/valor sao calculados em 64 bits para suportar isso; o resto
// do algoritmo original permanece identico.
// =========================================================

void construct(vector<long long>& msa, const vector<int>& loi, long long objectId, const Context& ctx) {

    for (int k = 0; k < ctx.n; k++) {

        long long idx = static_cast<long long>(k) * ctx.nn + objectId;

        msa[idx] = objectId * ctx.n + getPosition(k, loi);
    }
}

// =========================================================

void searchByObject(const vector<int>& lq, const vector<long long>& msa, const Context& ctx) {

    priority_queue<Node, vector<Node>, CompareMaxHeap> heap;

    for (int i = 0; i < ctx.nn; i++) {

        int score = 0;

        for (int k = 0; k < ctx.n; k++) {

            long long bucketStart = static_cast<long long>(lq[k]) * ctx.nn;

            int refPos = static_cast<int>(msa[bucketStart + i] % ctx.n);

            score += abs(refPos - k);
        }

        Node node(i, score);

        if ((int)heap.size() < KNN) {

            heap.push(node);

        } else {

            const Node& worst = heap.top();

            if (score < worst.d) {

                heap.pop();
                heap.push(node);
            }
        }
    }

    vector<Node> top;

    while (!heap.empty()) {

        top.push_back(heap.top());
        heap.pop();
    }

    sort(top.begin(), top.end(), [](const Node& a, const Node& b) {

             if (a.d != b.d)
                 return a.d < b.d;

             return a.i < b.i;
         });

    vector<int> result;

    for (const auto& n : top)
        result.push_back(n.i);

    printVector(result);
}

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

	cout << "msa_plain_int64.cpp" << endl;
	cout << "Programa: MSA original direto em RAM (versao com indices/msa em long long, sem armazenar em disco)" << endl;
    printMemory("Antes de comecar o programa");

    Context ctx;

    ifstream in(FILE_NAME);

    if (!in) {

        cerr << "Erro abrindo arquivo" << endl;

        return 1;
    }

    int totalObjs, totalRefs, totalQueries;

    in >> ctx.dim >> totalRefs >> totalObjs >> totalQueries;

    if (SEL_OBJS > totalObjs || SEL_REFS > totalRefs || SEL_QUES > totalQueries) {

        cerr << "Selecao maior que o disponivel no arquivo (objs=" << totalObjs
            << ", refs=" << totalRefs << ", queries=" << totalQueries << ")" << endl;

        return 1;
    }

    ctx.nn = SEL_OBJS;
    ctx.n = SEL_REFS;
    ctx.num_q = SEL_QUES;

    cout << "n = " << ctx.n << endl;
    cout << "nn = " << ctx.nn << endl;
    cout << "dim = " << ctx.dim << endl;
    cout << "num_q = " << ctx.num_q << endl;

    // msa agora e' vector<long long> (8 bytes por elemento)
    long long msaBytes = 8LL * ctx.n * ctx.nn;

    printf("MSA teorico = %.2f MB\n", msaBytes / 1024.0 / 1024.0);

    vector<double> r(static_cast<size_t>(ctx.dim) * ctx.n);

    // le apenas as primeiras SEL_REFS referencias do inicio da secao
    for (size_t i = 0; i < r.size(); i++) {

        in >> r[i];
    }

    // pula a secao de objetos inteira (total do arquivo, nao apenas os selecionados),
    // para que a secao de referencias comece sempre na mesma posicao do arquivo
    //skipTokens(in, static_cast<long long>(ctx.dim) * totalObjs);

    printMemory("Apos referencias");

    in.close();

    // =====================================================
    // SEGUNDA LEITURA
    // =====================================================

    ifstream in2(FILE_NAME);

    in2 >> ctx.dim >> totalRefs >> totalObjs >> totalQueries;

    ctx.nn = SEL_OBJS;
    ctx.n = SEL_REFS;
    ctx.num_q = SEL_QUES;

    vector<long long> msa(static_cast<size_t>(ctx.n) * static_cast<size_t>(ctx.nn));

    auto t1 = chrono::steady_clock::now();

    // pula a secao de referencias inteira (total do arquivo, nao apenas as
    // selecionadas), para que os objetos comecem sempre na mesma posicao do
    // arquivo, independente de SEL_REFS
    skipTokens(in2, static_cast<long long>(totalRefs) * ctx.dim);

    vector<double> oi(ctx.dim);

    for (long long i = 0; i < static_cast<long long>(ctx.dim) * ctx.nn; i++) {

        in2 >> oi[i % ctx.dim];

        if (i % ctx.dim == ctx.dim - 1) {

            vector<int> loi = getOrderedList(ctx, oi, r);

            construct(msa, loi, i / ctx.dim, ctx);
        }
    }

    auto t2 = chrono::steady_clock::now();

    printMemory("Apos construcao");

    cout << "TEMPO CONSTRUCAO = "
        << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count()
        << " ms" << endl;

    // pula o restante da secao de objetos que nao foi lido, para que as
    // queries comecem sempre na mesma posicao do arquivo, independente de
    // SEL_OBJS e SEL_REFS
    skipTokens(in2, static_cast<long long>(totalObjs - ctx.nn) * ctx.dim);

    // =====================================================
    // SEARCH
    // =====================================================

    for (int qid = 0; qid < ctx.num_q; qid++) {

        vector<double> q(ctx.dim);

        for (int j = 0; j < ctx.dim; j++) {

            in2 >> q[j];
        }

        vector<int> lq = getOrderedList(ctx, q, r);

        searchByObject(lq, msa, ctx);
    }

    auto t3 =
        chrono::steady_clock::now();

    printMemory(
        "Depois da busca");

    cout << "TEMPO BUSCA = "
        << chrono::duration_cast<chrono::milliseconds>(t3 - t2).count()
        << " ms" << endl;

    return 0;
}

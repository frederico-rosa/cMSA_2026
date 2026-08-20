// =============================================================
// msa_plain.cpp
//
// MSA original sem compressao, indices 32 bits (versao "base").
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
// CONFIGURA��ES
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
// MEM�RIA
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

void construct(vector<int>& msa, const vector<int>& loi, int objectId, const Context& ctx) {

    for (int k = 0; k < ctx.n; k++) {

        msa[k * ctx.nn + objectId] = objectId * ctx.n + getPosition(k, loi);
    }
}

// =========================================================

void searchByObject(const vector<int>& lq, const vector<int>& msa, const Context& ctx) {

    priority_queue<Node, vector<Node>, CompareMaxHeap> heap;

    for (int i = 0; i < ctx.nn; i++) {

        int score = 0;

        for (int k = 0; k < ctx.n; k++) {

            int bucketStart = lq[k] * ctx.nn;

            int refPos = msa[bucketStart + i] % ctx.n;

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

	cout << "msa_plain.cpp" << endl;
	cout << "Programa: MSA original direto em RAM (sem armazenar em disco)" << endl;
    printMemory("Antes de comecar o programa");

    Context ctx;

    ifstream in(FILE_NAME);

    if (!in) {

        cerr << "Erro abrindo arquivo" << endl;

        return 1;
    }

    int totalObjs, totalRefs, totalQueries;

    in >> ctx.dim >> totalObjs >> totalRefs >> totalQueries;

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

    long long msaBytes = 4LL * ctx.n * ctx.nn;

    printf("MSA teorico = %.2f MB\n", msaBytes / 1024.0 / 1024.0);

    vector<double> r(ctx.dim * ctx.n);

    double tmp;

    // pula a secao de objetos inteira (total do arquivo, nao apenas os selecionados),
    // para que a secao de referencias comece sempre na mesma posicao do arquivo
    for (long long i = 0; i < (long long)ctx.dim * totalObjs; i++) {

        in >> tmp;
    }

    // le apenas as primeiras SEL_REFS referencias do inicio da secao
    for (int i = 0; i < (int)r.size(); i++) {

        in >> r[i];
    }

    printMemory("Apos referencias");

    in.close();

    // =====================================================
    // SEGUNDA LEITURA
    // =====================================================

    ifstream in2(FILE_NAME);

    in2 >> ctx.dim >> totalObjs >> totalRefs >> totalQueries;

    ctx.nn = SEL_OBJS;
    ctx.n = SEL_REFS;
    ctx.num_q = SEL_QUES;

    vector<int> msa(ctx.n * ctx.nn);

    auto t1 = chrono::steady_clock::now();

    vector<double> oi(ctx.dim);

    for (int i = 0; i < ctx.dim * ctx.nn; i++) {

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

    // pula o restante da secao de objetos que nao foi lido
    for (long long i = 0; i < (long long)(totalObjs - ctx.nn) * ctx.dim; i++) {

        in2 >> tmp;
    }

    // pula a secao de referencias inteira (total do arquivo, nao apenas as
    // selecionadas), para que as queries comecem sempre na mesma posicao do
    // arquivo, independente de SEL_OBJS e SEL_REFS
    for (long long i = 0; i < (long long)totalRefs * ctx.dim; i++) {

        in2 >> tmp;
    }

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
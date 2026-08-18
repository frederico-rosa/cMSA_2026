// =============================================================
// cmsa_delta_pos.cpp
//
// Mesmo codec delta/gamma de cmsa_delta_gap.cpp, mas codificando a
// posicao diretamente (sem estado de diferenca entre objetos).
// =============================================================
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;

// ============================================================
// CONFIGURA��ES
// ============================================================

static const int KNN = 10;

// ============================================================
// CONTEXT
// ============================================================

struct Context
{

    int nn;
    int n;
    int dim;
    int num_q;

    vector<int> offset;
};

struct Node
{

    int i;
    double d;

    Node() : i(0), d(0.0)
    {
    }

    Node(int ii, double dd) : i(ii), d(dd)
    {
    }
};

// ============================================================
// MEMORIA
// ============================================================

long getVmRSSKB()
{
    ifstream f("/proc/self/status");

    string key;

    while (f >> key)
    {

        if (key == "VmRSS:")
        {

            long kb;
            string unit;

            f >> kb >> unit;

            return kb;
        }

        string line;
        getline(f, line);
    }

    return -1;
}

void printMemory(const string &msg)
{
    printf("%s : %.2f MB\n", msg.c_str(), getVmRSSKB() / 1024.0);
}

// ============================================================
// UTILIT�RIOS
// ============================================================

void print(const vector<int> &a)
{

    for (int v : a)
        cout << v << " ";

    cout << endl;
}

int getPosition(int ref, const vector<int> &loi)
{

    for (int i = 0; i < (int)loi.size(); i++)
    {

        if (loi[i] == ref)
            return i;
    }

    return -1;
}

// Calcula, em O(n) (uma unica passada sobre a lista ja ordenada), o
// mapeamento inverso pos[ref] = posicao(rank). Substitui n chamadas a
// getPosition() (cada uma O(n)) por n atribuicoes O(1), eliminando o
// custo O(n^2) por objeto que dominava a construcao para n grande.
vector<int> invertPositions(const Context &ctx, const vector<int> &loi)
{
    vector<int> pos(ctx.n);

    for (int rank = 0; rank < ctx.n; rank++)
        pos[loi[rank]] = rank;

    return pos;
}

// ============================================================
// CONTABILIDADE DELTA
// ============================================================

int writeBinaryCount(int value)
{

    int len = 0;

    while (value > 0)
    {

        value >>= 1;
        len++;
    }

    return len;
}

int gammaEncodeCount(int x)
{

    int len = 0;
    int temp = x;

    while (temp > 0)
    {

        temp >>= 1;
        len++;
    }

    int count = len - 1;

    count += writeBinaryCount(x);

    return count;
}

int deltaEncodeCount(int x)
{

    int len = 0;
    int temp = x;

    while (temp > 0)
    {

        temp >>= 1;
        len++;
    }

    return gammaEncodeCount(len) + (len - 1);
}

// ============================================================
// ESCRITA BIN�RIA
// ============================================================

void writeBinary(int value, int bucketID, vector<vector<uint8_t>> &buckets, vector<int> &offset)
{

    int len = 0;
    int temp = value;

    while (temp > 0)
    {

        temp >>= 1;
        len++;
    }

    for (int i = len - 1; i >= 0; i--)
    {

        if ((value >> i) & 1)
        {

            buckets[bucketID][offset[bucketID] / 8] |= (1u << (offset[bucketID] % 8));
        }

        offset[bucketID]++;
    }
}

// ============================================================
// GAMMA ENCODE
// ============================================================

void gammaEncode(int x, int bucketID, vector<vector<uint8_t>> &buckets, vector<int> &offset)
{

    int len = 0;
    int temp = x;

    while (temp > 0)
    {

        temp >>= 1;
        len++;
    }

    for (int i = 0; i < len - 1; i++)
    {

        offset[bucketID]++;
    }

    writeBinary(x, bucketID, buckets, offset);
}

// ============================================================
// DELTA ENCODE
// ============================================================

void deltaEncode(int x, int bucketID, vector<vector<uint8_t>> &buckets, vector<int> &offset)
{

    int len = 0;
    int temp = x;

    while (temp > 0)
    {

        temp >>= 1;
        len++;
    }

    gammaEncode(len, bucketID, buckets, offset);

    for (int i = len - 2; i >= 0; i--)
    {

        if ((x >> i) & 1)
        {

            buckets[bucketID][offset[bucketID] / 8] |= (1u << (offset[bucketID] % 8));
        }

        offset[bucketID]++;
    }
}

// ============================================================
// LEITURA DE BITS
// ============================================================

bool readBit(int bucketID, vector<vector<uint8_t>> &buckets, vector<int> &offset)
{

    uint8_t b = buckets[bucketID][offset[bucketID] / 8];

    int bit = (b >> (offset[bucketID] % 8)) & 1;

    offset[bucketID]++;

    return bit == 1;
}

// ============================================================
// DELTA DECODE
// ============================================================

int deltaDecode(int bucketID, vector<vector<uint8_t>> &buckets, vector<int> &offset)
{

    int zeros = 0;

    while (!readBit(bucketID, buckets, offset))
    {

        zeros++;
    }

    int len = 1;

    for (int i = 0; i < zeros; i++)
    {

        len = (len << 1) | (readBit(bucketID, buckets, offset) ? 1 : 0);
    }

    int value = 1 << (len - 1);

    for (int i = 0; i < len - 1; i++)
    {

        if (readBit(bucketID, buckets, offset))
        {

            value |= (1 << (len - i - 2));
        }
    }

    return value;
}

// =====================================================
// COUNT CODE SIZE
// =====================================================

void countCodeBitSize(const vector<int> &pos, Context &ctx, vector<int> &bucketBitSize)
{
    for (int k = 0; k < ctx.n; k++)
    {
        // Antes: codificava a DIFERENCA entre o valor absoluto deste
        // objeto (objectId*ctx.n+pos[k]) e o do objeto anterior no
        // mesmo bucket. Como pos varia em [0, ctx.n), essa diferenca
        // ficava tipicamente em torno de ctx.n, podendo chegar a
        // 2*ctx.n-1 no pior caso (quando a posicao "pula" de 0 para
        // ctx.n-1, ou vice-versa, entre dois objetos consecutivos).
        //
        // Agora: codifica a posicao diretamente (0..ctx.n-1), com bias
        // +1 exigido pelo codigo gamma/delta (que so' representa
        // valores >= 1). Valor maximo cai de ~2n para n — codigos, em
        // media, ~1 bit mais curtos — e nao ha' mais necessidade de
        // manter estado (firstElement/lastValue) entre objetos: cada
        // posicao e' autocontida.
        bucketBitSize[k] += deltaEncodeCount(pos[k] + 1);
    }
}

// =====================================================
// CONSTRUCT DELTA
// =====================================================

void constructDelta(vector<vector<uint8_t>> &buckets, vector<int> &offset, const vector<int> &pos, Context &ctx)
{
    for (int k = 0; k < ctx.n; k++)
    {
        deltaEncode(pos[k] + 1, k, buckets, offset);
    }
}

// =====================================================
// ORDERED LIST
// =====================================================

vector<int> getOrderedList(const Context &ctx, const vector<double> &oi, const vector<double> &r)
{
    vector<Node> loi(ctx.n);

    for (int k = 0; k < ctx.n; k++)
    {

        double dist = 0.0;

        for (int j = 0; j < ctx.dim; j++)
        {

            double term = oi[j] - r[k * ctx.dim + j];

            dist += term * term;
        }

        loi[k] = {k, dist};
    }

    sort(loi.begin(), loi.end(), [](const Node &a, const Node &b) {
        if (a.d != b.d)
            return a.d < b.d;

        return a.i < b.i;
    });

    vector<int> output(ctx.n);

    for (int i = 0; i < ctx.n; i++)
        output[i] = loi[i].i;

    return output;
}

// =====================================================
// SEARCH
// =====================================================

void searchByObject(const vector<int> &lq, vector<vector<uint8_t>> &buckets, vector<int> &offset, Context &ctx)
{
    auto cmp = [](const Node &a, const Node &b) {
        if (a.d != b.d)
            return a.d < b.d;

        return a.i < b.i;
    };

    priority_queue<Node, vector<Node>, decltype(cmp)> heap(cmp);

    for (int objectId = 0; objectId < ctx.nn; objectId++)
    {
        int score = 0;

        for (int k = 0; k < ctx.n; k++)
        {
            int bucketID = lq[k];

            // Cada posicao e' autocontida agora (nao ha' mais diferenca
            // acumulada entre objetos): basta decodificar e desfazer o
            // bias +1 usado na codificacao.
            int refPos = deltaDecode(bucketID, buckets, offset) - 1;

            score += abs(refPos - k);
        }

        Node node;
        node.i = objectId;
        node.d = score;

        if ((int)heap.size() < KNN)
        {
            heap.push(node);
        }
        else
        {
            Node worst = heap.top();

            if (score < worst.d)
            {
                heap.pop();
                heap.push(node);
            }
        }
    }

    vector<Node> top;

    while (!heap.empty())
    {
        top.push_back(heap.top());
        heap.pop();
    }

    sort(top.begin(), top.end(), [](const Node &a, const Node &b) {
        if (a.d != b.d)
            return a.d < b.d;

        return a.i < b.i;
    });

    vector<int> result;

    for (auto &n : top)
        result.push_back(n.i);

    print(result);
}

// =====================================================
// MAIN
// =====================================================

int main(int argc, char* argv[])
{

    if (argc < 5)
    {
        cerr << "Uso: " << argv[0] << " <SEL_OBJS> <SEL_REFS> <SEL_QUES> <arquivo_entrada>" << endl;
        return 1;
    }

    int SEL_OBJS = std::stoi(argv[1]);
    int SEL_REFS = std::stoi(argv[2]);
    int SEL_QUES = std::stoi(argv[3]);
    std::string FILE_NAME = argv[4];

	cout << "cmsa_delta_pos.cpp" << endl;
	cout << "Programa: cMSA otimizada delta direto em RAM (sem armazenar em disco)" << endl;
    printMemory("Antes de comecar o programa");

    Context ctx;

    // ==========================================
    // FIRST READ
    // ==========================================

    ifstream fin(FILE_NAME);

    if (!fin)
    {
        cerr << "Erro abrindo arquivo\n";
        return 1;
    }

    fin >> ctx.dim >> ctx.nn >> ctx.n >> ctx.num_q;

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

    double dummy;

    for (long long i = 0; i < (long long)ctx.dim * ctx.nn; i++)
    {
        fin >> dummy;
    }

    for (size_t i = 0; i < r.size(); i++)
    {
        fin >> r[i];
    }

    printMemory("Apos referencias");

    fin.close();

    // ==========================================
    // CONSTRUCTION
    // ==========================================

    auto t1 = chrono::high_resolution_clock::now();

    // ------------------------------------------
    // SECOND READ
    // COUNT SIZE
    // ------------------------------------------

    fin.open(FILE_NAME);

    fin >> ctx.dim >> ctx.nn >> ctx.n >> ctx.num_q;

    ctx.nn = SEL_OBJS;
    ctx.n = SEL_REFS;
    ctx.num_q = SEL_QUES;

    vector<int> bucketBitSize(ctx.n, 0);

    vector<double> oi(ctx.dim);

    for (long long i = 0; i < (long long)ctx.dim * ctx.nn; i++)
    {
        fin >> oi[i % ctx.dim];

        if ((i % ctx.dim) == ctx.dim - 1)
        {
            vector<int> loi = getOrderedList(ctx, oi, r);
            vector<int> pos = invertPositions(ctx, loi);

            countCodeBitSize(pos, ctx, bucketBitSize);
        }
    }

    fin.close();

    double cMsaBits = 0.0;

    for (int b : bucketBitSize)
        cMsaBits += b;

    printf("cMSA size = %.2f MB\n", cMsaBits / 8.0 / 1024.0 / 1024.0);

    // ------------------------------------------
    // THIRD READ
    // BUILD COMPRESSED CMSA
    // ------------------------------------------

    fin.open(FILE_NAME);

    fin >> ctx.dim >> ctx.nn >> ctx.n >> ctx.num_q;

    ctx.nn = SEL_OBJS;
    ctx.n = SEL_REFS;
    ctx.num_q = SEL_QUES;

    vector<vector<uint8_t>> buckets(ctx.n);

    for (int k = 0; k < ctx.n; k++)
    {
        buckets[k].resize(bucketBitSize[k] / 8 + 1, 0);
    }

    vector<int> offset(ctx.n, 0);

    for (long long i = 0; i < (long long)ctx.dim * ctx.nn; i++)
    {
        fin >> oi[i % ctx.dim];

        if ((i % ctx.dim) == ctx.dim - 1)
        {
            vector<int> loi = getOrderedList(ctx, oi, r);
            vector<int> pos = invertPositions(ctx, loi);

            constructDelta(buckets, offset, pos, ctx);
        }
    }

    auto t2 = chrono::high_resolution_clock::now();

    printMemory("Apos construcao");

    cout << "TEMPO CONSTRUCAO = " << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << " ms\n";

    // ==========================================
    // SKIP REFERENCES
    // ==========================================

    for (size_t i = 0; i < r.size(); i++)
    {
        fin >> dummy;
    }

    // ==========================================
    // SEARCH
    // ==========================================

    for (int qid = 0; qid < ctx.num_q; qid++)
    {
        fill(offset.begin(), offset.end(), 0);

        vector<double> q(ctx.dim);

        for (int j = 0; j < ctx.dim; j++)
        {
            fin >> q[j];
        }

        vector<int> lq = getOrderedList(ctx, q, r);

        searchByObject(lq, buckets, offset, ctx);
    }

    fin.close();

    auto t3 = chrono::high_resolution_clock::now();

    printMemory("Depois da busca");

    cout << "TEMPO BUSCA = " << chrono::duration_cast<chrono::milliseconds>(t3 - t2).count() << " ms\n";

    return 0;
}
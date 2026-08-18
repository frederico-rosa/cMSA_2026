// =============================================================
// cmsa_delta_gap.cpp
//
// cMSA comprimida com codigos Elias delta/gamma, codificando o
// gap (diferenca) entre objetos consecutivos no bucket.
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

void countCodeBitSize(const vector<int> &pos, int objectId, Context &ctx, vector<bool> &firstElement,
                      vector<int> &lastValue, vector<int> &bucketBitSize)
{
    for (int k = 0; k < ctx.n; k++)
    {
        // objectId*ctx.n pode passar de 2^31 (ex.: objectId~1M, n=5000).
        // Calculamos em 64 bits e truncamos com seguranca (definido pelo
        // padrao), evitando o overflow de int*int, que e' UB.
        long long value64 = static_cast<long long>(objectId) * ctx.n + pos[k];
        int value = static_cast<int>(value64);

        if (firstElement[k])
        {
            firstElement[k] = false;
            lastValue[k] = value;

            bucketBitSize[k] += deltaEncodeCount(value + 1);
        }
        else
        {
            int diff = value - lastValue[k];

            lastValue[k] = value;

            bucketBitSize[k] += deltaEncodeCount(diff);
        }
    }
}

// =====================================================
// CONSTRUCT DELTA
// =====================================================

void constructDelta(vector<vector<uint8_t>> &buckets, vector<int> &offset, const vector<int> &pos, int objectId,
                    Context &ctx, vector<bool> &firstElement, vector<int> &lastValue)
{
    for (int k = 0; k < ctx.n; k++)
    {
        long long value64 = static_cast<long long>(objectId) * ctx.n + pos[k];
        int value = static_cast<int>(value64);

        if (firstElement[k])
        {
            firstElement[k] = false;
            lastValue[k] = value;

            deltaEncode(value + 1, k, buckets, offset);
        }
        else
        {
            int diff = value - lastValue[k];

            lastValue[k] = value;

            deltaEncode(diff, k, buckets, offset);
        }
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

void searchByObject(const vector<int> &lq, vector<vector<uint8_t>> &buckets, vector<int> &offset,
                    vector<bool> &firstElement, vector<int> &lastValue, Context &ctx)
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

            int refPos;

            if (firstElement[k])
            {
                int msaValue = (deltaDecode(bucketID, buckets, offset) - 1) % ctx.n;

                lastValue[k] = msaValue;
                firstElement[k] = false;

                refPos = msaValue;
            }
            else
            {
                int decodedValue = deltaDecode(bucketID, buckets, offset);

                int msaValue = decodedValue + lastValue[k];

                lastValue[k] = msaValue;

                // msaValue reconstroi objectId*ctx.n+pos[k] truncado em 32
                // bits; uma vez que objectId*ctx.n+pos[k] ultrapassa
                // 2^31-1, o valor truncado vira negativo (reinterpretacao
                // em complemento de dois) e o operador % do C++ (que
                // trunca em direcao a zero) devolve resto negativo em vez
                // do modulo euclidiano esperado. Normaliza para [0, ctx.n).
                refPos = msaValue % ctx.n;
                if (refPos < 0)
                {
                    refPos += ctx.n;
                }
            }

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

	cout << "cmsa_delta_gap.cpp" << endl;
	cout << "Programa: cMSA delta direto em RAM (sem armazenar em disco)" << endl;
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

    vector<bool> firstElement(ctx.n, true);

    vector<int> lastValue(ctx.n, 0);

    vector<int> bucketBitSize(ctx.n, 0);

    vector<double> oi(ctx.dim);

    for (long long i = 0; i < (long long)ctx.dim * ctx.nn; i++)
    {
        fin >> oi[i % ctx.dim];

        if ((i % ctx.dim) == ctx.dim - 1)
        {
            vector<int> loi = getOrderedList(ctx, oi, r);
            vector<int> pos = invertPositions(ctx, loi);

            countCodeBitSize(pos, (int)(i / ctx.dim), ctx, firstElement, lastValue, bucketBitSize);
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

    fill(firstElement.begin(), firstElement.end(), true);

    fill(lastValue.begin(), lastValue.end(), 0);

    for (long long i = 0; i < (long long)ctx.dim * ctx.nn; i++)
    {
        fin >> oi[i % ctx.dim];

        if ((i % ctx.dim) == ctx.dim - 1)
        {
            vector<int> loi = getOrderedList(ctx, oi, r);
            vector<int> pos = invertPositions(ctx, loi);

            constructDelta(buckets, offset, pos, (int)(i / ctx.dim), ctx, firstElement, lastValue);
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

        fill(firstElement.begin(), firstElement.end(), true);

        fill(lastValue.begin(), lastValue.end(), 0);

        vector<double> q(ctx.dim);

        for (int j = 0; j < ctx.dim; j++)
        {
            fin >> q[j];
        }

        vector<int> lq = getOrderedList(ctx, q, r);

        searchByObject(lq, buckets, offset, firstElement, lastValue, ctx);
    }

    fin.close();

    auto t3 = chrono::high_resolution_clock::now();

    printMemory("Depois da busca");

    cout << "TEMPO BUSCA = " << chrono::duration_cast<chrono::milliseconds>(t3 - t2).count() << " ms\n";

    return 0;
}
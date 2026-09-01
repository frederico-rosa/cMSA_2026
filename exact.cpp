// =============================================================
// exact.cpp: compute exact KNN queries using sequential scan
// =============================================================
//#include <algorithm>
#include <cstdlib>
#include <cstdio>
//#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
//#include <queue>
#include <string>
//#include <vector>
#include <map>

using namespace std;

//#define MANHATTAN
#define EUCLIDEAN

// =========================================================
// CONFIGURACOES
// =========================================================
static const int KNN = 10;

// =========================================================
// CONTEXT
// =========================================================
struct Context {
    int nn; // number of records
    int n;  // number of references
    int dim; // number of dimensions
    int num_q; // number of queries
};

float distance(double *o1, double *o2, int dim) {
    float d = 0;
    #ifdef MANHATTAN
    for (int i = 0; i < dim; i++) d += fabs(o1[i] - o2[i]);
    #endif
    #ifdef EUCLIDEAN
    for (int i = 0; i < dim; i++) 
        d += (o1[i] - o2[i]) * (o1[i] - o2[i]);
    d = sqrt(d);
    #endif
    return d;
}

// red black tree to index pairs of < distance, offset >
class result {
  private:
    typedef multimap<float, int> mapType; // multimap allows duplicate keys, maps ignores duplicate keys
    mapType tree;
    mapType::iterator iter;
    float maxkey;
  public:

    result() { maxkey = -1; }
    
    ~result() { tree.clear(); }

    void add(float key, int value) { tree.insert(pair<float, int>(key, value)); if (key > maxkey) maxkey = key; }

    void remove(float key) {
        iter = tree.find(key);
        if (iter != tree.end()) {
            tree.erase(iter);
        }
    }

    void removelast() { 
        iter = tree.end();
        --iter;
        tree.erase(iter);
        iter = tree.end();
        --iter;
        maxkey = getkey();
    }
    
    void clear() { tree.clear(); }

    void prior() { --iter; }

    void next() { ++iter; }

    bool eof() { if (iter == tree.end()) return true; return false; }

    bool bof() { if (iter == tree.begin()) return true; return false; }

    float getkey() { return iter->first; }

    int getvalue() { return iter->second; }
    
    float getmaxkey() { return maxkey; } 

    void begin() { iter = tree.begin(); }

    void end() { iter = tree.end(); }

    int find(float key) {
        iter = tree.find(key);
        if (iter != tree.end()) {
            return iter->second;
        }
        else {
            return -1;
        }
    }

    int count() { return tree.size(); }
};

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

    //cout << "Program: exact KNN queries, memory based sequential scan" << endl;

    Context ctx;

    ifstream in(FILE_NAME);

    if (!in) {
        cerr << "Could not open " << FILE_NAME << endl;
        return 1;
    }

    in >> ctx.dim      // num dimensions
       >> ctx.n        // num references
       >> ctx.nn       // num records
       >> ctx.num_q;   // num queries

    //cout << "nn = " << ctx.nn << " (num records)" << endl; // num records
    //cout << "n = " << ctx.n << " (num references)" << endl; // num references
    //cout << "dim = " << ctx.dim << endl; // dim
    //cout << "num_q = " << ctx.num_q << " (num queries)" << endl; // num queries

    float tmp;

    // ignore references
    for (int i = 0; i < ctx.dim * ctx.n; i++) {
        in >> tmp;
    }

    // ignore records
    for (int i = 0; i < ctx.dim * ctx.nn; i++) {
        in >> tmp;
    }

    // read queries
    double **queries = new double*[ctx.num_q];
    for (int i = 0; i < ctx.num_q ; i++) {
        queries[i] = new double[ctx.dim];
        for (int j = 0; j < ctx.dim; j++) { 
            in >> queries[i][j]; 
        }
    }

    in.close();

    // =====================================================
    // SECOND READ
    // =====================================================

    ifstream in2(FILE_NAME);

    if (!in2) {
        cerr << "Could not open " << FILE_NAME << endl;
        return 1;
    }

    in2 >> tmp;
    in2 >> tmp;
    in2 >> tmp;
    in2 >> tmp;

    // ignore references
    for (int i = 0; i < ctx.dim * ctx.n; i++) {
        in2 >> tmp;
    }

    //ctx.n = SEL_REFS;
    ctx.nn = SEL_OBJS;    // let's consider the number of records the user wants
    ctx.num_q = SEL_QUES; // let's consider the number of queries the user wants

    // read objects: for each object, consider it to be included in the result of each query
    result *results = new result[ctx.num_q];
    double *obj = new double[ctx.dim];
    for (int i = 0; i < ctx.nn; i++) {
        // read 1 object
        for (int kk = 0; kk < ctx.dim; kk++) { in2 >> obj[kk]; }
            
        // for each query
        for (int j = 0; j < ctx.num_q; j++) {
            float max, d = distance(queries[j], obj, ctx.dim);
            // add if count < k
            if (results[j].count() < KNN) {
                results[j].add(d, i);
            }
            // otherwise, add if distance is smaller than maxkey and remove last
            else {
                max = results[j].getmaxkey();
                if (d < max) {
                    results[j].add(d, i);
                    results[j].removelast();
                }
            }
        }           
    }
    delete[]obj;
    // for each query
    for (int i = 0; i < ctx.num_q; i++) {
        results[i].begin();
	int key;
	while (!results[i].eof()) {
	    key = results[i].getvalue();
	    cout << key << " ";
	    results[i].next();
	}
	cout << endl;
    }

    return 0;
}


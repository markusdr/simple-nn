// Markus Dreyer, Feb 2015
//
// A neural network for tagging (experimental)

#include <iostream>
#include <iomanip>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unistd.h> // getopt

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <adept.h>

#define DBG(x) do{ std::stringstream ss; ss << x; std::cerr << ss.str() << std::endl; } while (0)
#define THROW(x) do{ std::stringstream ss; ss << x; throw std::runtime_error(ss.str()); } while (0)

typedef adept::adouble AdeptDouble;

template<class T>
T getRandom(T const& min, T const& max) {
  return min + rand() / (RAND_MAX / (max - min));
}

/**
 * Provides string->id mapping
 */
class Vocabulary {
public:
  typedef std::size_t Id;
  typedef std::unordered_map<std::string, Id> Map;

  Vocabulary(std::string const& fname) {
    std::ifstream in(fname);
    if (!in.is_open()) THROW("Could not read '" << fname << "'");
    DBG("Reading '" << fname << "'");
    Id id = 0;
    std::string word;
    while (in >> word) {
      map_[word] = id++;
    }
    doc_begin_ = map_["<s>"];
    doc_end_ = map_["</s>"];
    unk_ = map_["UUUNKKK"];
  }

  Id size() const {
    return map_.size();
  }

  Id GetId(std::string const& word) const {
    Map::const_iterator iter = map_.find(word);
    return iter == map_.end() ? unk_ : iter->second;
  }

  Id doc_begin_, doc_end_, unk_;

private:
  Map map_;
};

/**
 * Options for neural network
 */
struct Options {
  Options(int argc, char** argv)
    : nonlinear_fct("tanh"),
      hidden_layer_dim(25),
      window_size(5),
      data_dir("data"),
      skip_test(false),
      learn_rate(0.01)
  {
    int c;
    while ((c = getopt (argc, argv, "d:f:H:n:l:th")) != -1)
      switch (c) {
      case 'd':
        data_dir = std::string(optarg);
        break;
      case 'f':
        nonlinear_fct = std::string(optarg);
        break;
      case 'H':
        hidden_layer_dim = atoi(optarg);
        break;
      case 'n':
        window_size = atoi(optarg);
        break;
      case 'l':
        learn_rate = atof(optarg);
        break;
      case 't':
        skip_test = true;
        break;
      case 'h':
        std::cout << argv[0] << ": A neural network for tagging (experimental).\nOptions:\n"
                  <<" -h      Show this help text.\n"
                  <<" -d dir  Specify the data directory.\n"
                  <<" -f [tanh|sigmoid|softplus] Specify the nonlinear function.\n"
                  <<" -n num  Specify the window size.\n"
                  <<" -H num  Specify the hidden layer size.\n"
                  <<" -l num  Specify the gradient descent learning rate.\n"
                  <<" -s      Skip test and evaluation at the end.\n"
          ;
        exit(1);
      default:
        THROW("Options error, see -h");
      }

    if (nonlinear_fct != "tanh" && nonlinear_fct != "sigmoid" && nonlinear_fct != "softplus") {
      THROW("Nonlinear function '" << nonlinear_fct << "' not implemented");
    }

    DBG("Running: "
        << argv[0] << " -f " << nonlinear_fct << " -h " << hidden_layer_dim << " -n " << window_size
        << " -l " << learn_rate << " -d \"" << data_dir << "\"" << (skip_test ? " -s" : ""));
  }

  std::string nonlinear_fct;
  std::size_t hidden_layer_dim;
  std::size_t window_size;
  std::string data_dir;
  bool skip_test;
  double learn_rate;
};

template<class T>
T sigmoid(T const& x) {
  return 1.0 / (1.0 + exp(-x));
}

struct TanhFct {
  template<class T>
  T operator()(T const& f) { return tanh(f); }
};

struct SigmoidFct {
  template<class T>
  T operator()(T const& f) { return sigmoid(f); }
};

// Soft plus (soft rectified linear unit)
struct SoftplusFct {
  template<class T>
  T operator()(T const& f) { return log(1.0 + exp(f)); }
};

template<class NonlinearFct>
class NeuralNet {

public:
  typedef Eigen::Matrix<AdeptDouble, Eigen::Dynamic, Eigen::Dynamic > AdeptMatrix;
  typedef std::vector<bool> Labels;

  NeuralNet(Options const& opts) : opts_(opts) {
    srand(static_cast<unsigned>(time(0)));

    std::string vocab_file = opts.data_dir + "/vocab.txt";
    std::string vectors_file = opts.data_dir + "/wordVectors.txt";
    std::string train_file = opts.data_dir + "/train";

    vocab = new Vocabulary(vocab_file);

    ReadWordVectors(vectors_file, &word_vectors);

    context_len = opts.window_size / 2;

    for (std::size_t i = 0; i < context_len; ++i) {
      train_sequence.push_back(vocab->doc_begin_);
    }
    ReadLabeledSequence(train_file, &train_sequence, &train_labels, *vocab);
    for (std::size_t i = 0; i < context_len; ++i) {
      train_sequence.push_back(vocab->doc_end_);
    }

    input.resize(opts.window_size * word_vec_dim + 1,  1); // row vector

    InitParams();
  }

  void InitParams() {
    std::size_t input_size = opts_.window_size * word_vec_dim + 1;

    W.resize(opts_.hidden_layer_dim, input_size);
    float max = 1.0 / sqrt((float)input_size);
    for (std::size_t i = 0; i < opts_.hidden_layer_dim; ++i) {
      for (std::size_t j = 0; j < input_size; ++j) {
        W(i, j) = getRandom(-max, max);
      }
    }

    V.resize(1, opts_.hidden_layer_dim);
    for (std::size_t i = 0; i < opts_.hidden_layer_dim; ++i) {
      V(0, i) = getRandom(-max, max);
    }
  }

  ~NeuralNet() {
    delete vocab;
  }

  void Test() {
    std::vector<Vocabulary::Id> test_sequence;
    for (std::size_t i = 0; i < context_len; ++i) {
      test_sequence.push_back(vocab->doc_begin_);
    }
    Labels test_labels;
    std::string test_file = opts_.data_dir + "/dev";
    ReadLabeledSequence(test_file, &test_sequence, &test_labels, *vocab);
    for (std::size_t i = 0; i < context_len; ++i) {
      test_sequence.push_back(vocab->doc_end_);
    }

    DBG("Testing");
    unsigned tp = 0, fp = 0, tn = 0, fn = 0;

    s.new_recording(); // Avoids memory growth. TODO: We don't need adept at all here.
    s.pause_recording();

    // Now use float instead of AdeptDouble:
    W_float.resize(W.rows(), W.cols());
    for (std::size_t r = 0; r < W.rows(); ++r) {
      for (std::size_t c = 0; c < W.cols(); ++c) {
        W_float(r, c) = W(r, c).value();
      }
    }
    V_float.resize(V.rows(), V.cols());
    for (std::size_t r = 0; r < V.rows(); ++r) {
      for (std::size_t c = 0; c < V.cols(); ++c) {
        V_float(r, c) = V(r, c).value();
      }
    }
    input_float.resize(input.rows(),  1);

    for (std::size_t i = 0; i < test_sequence.size() - 2 * context_len; ++i ) {
      GetInputVecFloat(test_sequence, i, word_vectors, &input_float);
      float score = ComputeScore(W_float, V_float, input_float);
      if (score > 0.5 && test_labels[i]) {
        tp++;
      } else if (score > 0.5 && !test_labels[i]) {
        fp++;
      } else if (score <= 0.5 && !test_labels[i]) {
        tn++;
      } else if (score <= 0.5 && test_labels[i]) {
        fn++;
      }
    }

    float prec = (float)tp / (tp + fp);
    float rec = (float)tp / (tp + fn);
    float f1 = (float) 2.0 * prec * rec / (prec + rec);
    std::cout << "Test: Precision=" << tp << "/" << (tp + fp) << "=" << prec
              << ", Recall=" << tp << "/" << (tp + fn) << "=" << rec << ", F1=" << f1 << std::endl;
  }

  /**
   * Simple gradient descent training
   */
  void Train() {
    DBG("Training");
    for (std::size_t i = 0; i < train_sequence.size() - 2 * context_len; ++i ) {
    // for (std::size_t i = 0; i < 10000; ++i ) { // faster test
      if ((i+1) % 10000 == 0) {
        DBG(i+1);
      }

      s.new_recording();

      GetInputVec(train_sequence, i, word_vectors, &input);
      AdeptDouble loss = ComputeLoss(W, V, input, train_labels[i]);

      loss.set_gradient(1.0);
      s.compute_adjoint();

      // Update parameters (get_gradient is provided by the adept library.)
      for (std::size_t r = 0; r < W.rows(); ++r) {
        for (std::size_t c = 0; c < W.cols(); ++c) {
          W(r, c) -= opts_.learn_rate * W(r, c).get_gradient();
        }
      }
      for (std::size_t c = 0; c < opts_.hidden_layer_dim; ++c) {
        V(0, c) -= opts_.learn_rate * V(0, c).get_gradient();
      }
      std::set<Vocabulary::Id> word_ids; // removes duplicate words in window
      for (std::size_t r = 0; r < opts_.window_size; ++r) {
	word_ids.insert(train_sequence[i + r]);
      }
      for (std::set<Vocabulary::Id>::const_iterator it = word_ids.begin();
	   it != word_ids.end(); ++it) {
        for (std::size_t c = 0; c < word_vec_dim; ++c) {
	  // use smaller learning rate here because word vectors have good initialization
          word_vectors(*it, c) -= 0.1 * opts_.learn_rate * word_vectors(*it, c).get_gradient();
        }
      }
    }
  }

private:
  std::size_t GetNumTokens(std::string const& str) const {
    std::stringstream ss(str);
    std::size_t n = 0;
    float f;
    while (ss >> f) {
      ++n;
    }
    return n;
  }

  void ReadWordVectors(std::string const& fname,
                       AdeptMatrix* vectors) {
    std::ifstream in(fname);
    if (!in.is_open()) THROW("Could not read '" << fname << "'");
    float f;
    DBG("Reading '" << fname << "'");
    std::string line;
    for (Vocabulary::Id r = 0; r < vocab->size(); ++r) {
      std::getline(in, line);
      if (r == 0) {
        std::size_t num_floats = GetNumTokens(line);
        DBG("Found word vectors dimension: " << num_floats);
        word_vectors.resize(vocab->size(), num_floats);
        word_vec_dim = num_floats;
      }
      std::stringstream ss(line);
      std::size_t c = 0;
      while (ss >> f) {
        (*vectors)(r, c++) = f;
      }
      if (c != word_vec_dim) {
        THROW("Wrong dimension in '" << line << "'");
      }
    }
  }

  /**
   * Reads labeled training or test data
   */
  void ReadLabeledSequence(std::string const& fname,
                           std::vector<Vocabulary::Id>* ids,
                           Labels* labels,
                           Vocabulary const& vocab) {
    std::ifstream in(fname);
    if (!in.is_open()) THROW("Could not read '" << fname << "'");
    std::string word, label;
    DBG("Reading '" << fname << "'");
    while (in >> word >> label) {
      ids->push_back(vocab.GetId(word));
      labels->push_back(label != "O");
    }
  }

  /**
   * Computes neural network score (between 0 and 1) for one input
   * window.
   */
  template<class Matrix>
  typename Matrix::Scalar ComputeScore(Matrix const& W,
                                       Matrix const& V,
                                       Matrix const& input) {
    Matrix h(W * input); // slow when run with AdeptDouble matrix

    for (std::size_t i = 0; i < opts_.hidden_layer_dim; ++i) {
      h(i, 0) = nonlinear_fct(h(i, 0));
    }
    return sigmoid((V * h)(0,0));
  }

  /*
   * Computes loss based on the correct label 0 or 1.
   */
  template<class Matrix>
  typename Matrix::Scalar ComputeLoss(Matrix const& W,
                                      Matrix const& V,
                                      Matrix const& input,
                                      bool label) {
    AdeptDouble score = ComputeScore(W, V, input);
    float label_f = (float)label;
    return -label_f * log(score) - (1.0 - label_f) * log(1.0 - score);
  }

  /**
   * Copies all word vectors of the window into one row vectors
   */
  template<class Input>
  void GetInputVec(std::vector<Vocabulary::Id> const& seq,
                   std::size_t i,
                   AdeptMatrix const& word_vectors,
                   Input* input) {
    std::size_t row = 0;
    for (std::size_t n = 0; n < opts_.window_size; ++n) {
      for (std::size_t j = 0; j < word_vec_dim; ++j) {
        (*input)(row++, 0) = word_vectors(seq[i + n], j);
      }
    }
    (*input)(row, 0) = 1.0; // bias
  }

  /**
   * Same as GetInputVec, but uses the float value(), not AdeptDouble.
   */
  template<class Input>
  void GetInputVecFloat(std::vector<Vocabulary::Id> const& seq,
                        std::size_t i,
                        AdeptMatrix const& word_vectors,
                        Input* input) {
    std::size_t row = 0;
    for (std::size_t n = 0; n < opts_.window_size; ++n) {
      for (std::size_t j = 0; j < word_vec_dim; ++j) {
        (*input)(row++, 0) = word_vectors(seq[i + n], j).value();
      }
    }
    (*input)(row, 0) = 1.0; // bias
  }

  Options opts_;

  adept::Stack s;

  Vocabulary* vocab;
  std::size_t context_len;
  std::vector<Vocabulary::Id> train_sequence;
  Labels train_labels;
  std::size_t word_vec_dim;
  NonlinearFct nonlinear_fct;

  AdeptMatrix word_vectors, input, W, V;
  Eigen::MatrixXf input_float, W_float, V_float; // MatrixXf: dynamic-size float matrix
};

template<class NonlinearFct>
void runNeuralNet(Options const& opts) {
  NeuralNet<NonlinearFct> nn(opts);
  nn.Train();
  if (!opts.skip_test)
    nn.Test();
}

int main(int argc, char** argv) {
  Options opts(argc, argv);

  if (opts.nonlinear_fct == "tanh") {
    runNeuralNet<TanhFct>(opts);
  }
  else if (opts.nonlinear_fct == "sigmoid") {
    runNeuralNet<SigmoidFct>(opts);
  }
  else if (opts.nonlinear_fct == "softplus") {
    runNeuralNet<SoftplusFct>(opts);
  }

  return EXIT_SUCCESS;
}

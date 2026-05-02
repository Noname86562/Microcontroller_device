// host_test_cpp_stream.cpp
// Build example:
// ORT_DIR=onnxruntime-linux-x64-1.15.1
// g++ host_test_cpp_stream.cpp -o host_test_cpp_stream \
//   -I"${ORT_DIR}/include" -L"${ORT_DIR}/lib" -lonnxruntime \
//   -Wl,-rpath,"${ORT_DIR}/lib" -O2 -std=c++17
//
// Usage (batch):   ./host_test_cpp_stream model.onnx feat.bin
// Usage (stream):  ./host_test_cpp_stream encoder.onnx head.onnx feat.bin

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <memory>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cmath>

using namespace std;
using hrclock = std::chrono::high_resolution_clock;

static vector<float> load_bin_features(const string &path, int &T, int &F) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("Cannot open " + path);
    int32_t t32 = 0, f32 = 0;
    in.read(reinterpret_cast<char*>(&t32), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&f32), sizeof(int32_t));
    if (!in) throw runtime_error("Failed reading header");
    T = (int)t32; F = (int)f32;
    if (T <= 0 || F <= 0) throw runtime_error("Invalid dims");
    size_t nelems = size_t(T) * size_t(F);
    vector<float> data(nelems);
    in.read(reinterpret_cast<char*>(data.data()), sizeof(float) * nelems);
    if (!in) throw runtime_error("Failed reading data");
    return data;
}
// minimal .npy v1.x loader (assumes little-endian float32 contiguous, shape tuple present)
static vector<float> load_npy(const string &path, int &T, int &F) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("Cannot open " + path);
    // read magic
    char magic[6];
    in.read(magic, 6);
    if (strncmp(magic, "\x93NUMPY", 6) != 0) throw runtime_error("Not a .npy file");
    unsigned char ver[2];
    in.read(reinterpret_cast<char*>(ver), 2);
    // header length (little-endian 2-byte for v1.x)
    uint16_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), 2);
    string header;
    header.resize(header_len);
    in.read(&header[0], header_len);
// find shape
    auto p1 = header.find("shape");
    if (p1 == string::npos) throw runtime_error("Failed to parse header (no shape)");
    auto paren1 = header.find('(', p1);
    auto paren2 = header.find(')', paren1);
    if (paren1 == string::npos || paren2 == string::npos) throw runtime_error("Failed to parse shape");
    string shape_str = header.substr(paren1+1, paren2-paren1-1);
    // parse ints
    vector<int> dims;
    {
        size_t pos = 0;
        while (pos < shape_str.size()) {
            while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ',')) pos++;
            if (pos >= shape_str.size()) break;
            size_t end = pos;
            while (end < shape_str.size() && (isdigit((unsigned char)shape_str[end]) || shape_str[end]=='-')) end++;
            if (end==pos) break;
            int v = stoi(shape_str.substr(pos, end-pos));
            dims.push_back(v);
            pos = end;
        }
 }
    if (dims.size() < 2) throw runtime_error("Npy shape must be 2D (T,F)");
    T = dims[0];
    F = dims[1];
    // read rest as float32
    size_t nelems = size_t(T) * size_t(F);
    vector<float> data(nelems);
    in.read(reinterpret_cast<char*>(data.data()), sizeof(float) * nelems);
    if (!in) throw runtime_error("Failed reading npy data");
    return data;
}
static vector<float> load_features_any(const string &path, int &T, int &F) {
    string ext;
    auto p = path.find_last_of('.');
    if (p != string::npos) ext = path.substr(p+1);
    if (ext == "bin") return load_bin_features(path, T, F);
    if (ext == "npy") return load_npy(path, T, F);
    throw runtime_error("Unsupported feature file extension: " + path);
}


int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 4) {
            cerr << "Usage (batch):   " << argv[0] << " model.onnx feat.(bin|npy)\n";
            cerr << "Usage (stream):  " << argv[0] << " encoder.onnx head.onnx feat.(bin|npy)\n";
            return 1;
        }

        bool streaming_mode = (argc == 4);
        string model_a = argv[1];
        string model_b = streaming_mode ? argv[2] : string();
        string feat_path = streaming_mode ? argv[3] : argv[2];
  int T=0, F=0;
        auto data = load_features_any(feat_path, T, F);
        cout << "[info] Loaded features: T=" << T << " F=" << F << " total=" << data.size() << "\n";

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "host_test");
        Ort::SessionOptions sess_opts;
        sess_opts.SetIntraOpNumThreads(1);
        sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
if (!streaming_mode) {
            // Batch whole-model inference
            Ort::Session session(env, model_a.c_str(), sess_opts);
            Ort::AllocatorWithDefaultOptions allocator;

            // get input name(s)
            size_t num_inputs = session.GetInputCount();
            if (num_inputs < 1) throw runtime_error("Model has no inputs");
            Ort::AllocatedStringPtr iname_alloc = session.GetInputNameAllocated(0, allocator);
            const char* iname = iname_alloc.get();

            // create input tensor shape (1,T,F)
            vector<int64_t> input_shape = {1, (int64_t)T, (int64_t)F};
            size_t input_size = size_t(1) * size_t(T) * size_t(F);

            Ort::MemoryInfo meminfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(meminfo, data.data(), input_size, input_shape.data(), input_shape.size());

            // outputs (keep allocated name objects alive)
            size_t out_count = session.GetOutputCount();
            vector<Ort::AllocatedStringPtr> out_allocs;
            vector<const char*> out_names;
            out_allocs.reserve(out_count);
            out_names.reserve(out_count);
            for (size_t i=0;i<out_count;++i) {
                out_allocs.push_back(session.GetOutputNameAllocated(i, allocator));
                out_names.push_back(out_allocs.back().get());
            }
 vector<const char*> in_names = { iname };

            auto t0 = hrclock::now();
            auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names.data(), &input_tensor, 1, out_names.data(), out_names.size());
            auto t1 = hrclock::now();

            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            cout << "[timing] model (batch) total ms: " << elapsed_ms << "\n";

            // assume first output is logits (1, nclasses)
            assert(outputs.size() >= 1);
            float* outptr = outputs[0].GetTensorMutableData<float>();
            auto info = outputs[0].GetTensorTypeAndShapeInfo();
            vector<int64_t> outshape = info.GetShape();
            int nclasses = (int)outshape.back();
            vector<float> logits(outptr, outptr + nclasses);

  // softmax using std::exp
            float m = *max_element(logits.begin(), logits.end());
            float ssum = 0.0f;
            for (float &v : logits) { v = std::exp(v - m); ssum += v; }
            for (float &v : logits) v /= ssum;
            int best = max_element(logits.begin(), logits.end()) - logits.begin();
            cout << "[batch] Top1: " << best << " prob=" << logits[best] << "\n";

            return 0;
        } else {
            // Streaming mode: encoder.onnx, head.onnx
            string encoder_path = model_a;
            string head_path = model_b;
            Ort::Session enc_sess(env, encoder_path.c_str(), sess_opts);
            Ort::Session head_sess(env, head_path.c_str(), sess_opts);
            Ort::AllocatorWithDefaultOptions allocator;

// Get encoder inputs/outputs names and shapes (keep allocated names alive)
            size_t enc_in_ct = enc_sess.GetInputCount();
            if (enc_in_ct < 1) throw runtime_error("Encoder ONNX must have at least one input (x_t)");
            vector<Ort::AllocatedStringPtr> enc_in_allocs;
            vector<string> enc_in_names;
            vector<vector<int64_t>> enc_in_shapes;
            enc_in_allocs.reserve(enc_in_ct);
            enc_in_names.reserve(enc_in_ct);
            for (size_t i=0;i<enc_in_ct;++i) {
                enc_in_allocs.push_back(enc_sess.GetInputNameAllocated(i, allocator));
                enc_in_names.push_back(string(enc_in_allocs.back().get()));
                Ort::TypeInfo ti = enc_sess.GetInputTypeInfo(i);
                auto tensor_info = ti.GetTensorTypeAndShapeInfo();
                enc_in_shapes.push_back(tensor_info.GetShape());
            }
 // outputs
            size_t enc_out_ct = enc_sess.GetOutputCount();
            vector<Ort::AllocatedStringPtr> enc_out_allocs;
            vector<string> enc_out_names;
            enc_out_allocs.reserve(enc_out_ct);
            enc_out_names.reserve(enc_out_ct);
            for (size_t i=0;i<enc_out_ct;++i) {
                enc_out_allocs.push_back(enc_sess.GetOutputNameAllocated(i, allocator));
                enc_out_names.push_back(string(enc_out_allocs.back().get()));
            }
 // Build initial zero states for enc_inputs[1:]
            vector<vector<float>> state_storage; // flattened buffers per state
            vector<vector<int64_t>> state_shapes;
            for (size_t i=1;i<enc_in_ct;++i) {
                vector<int64_t> s = enc_in_shapes[i];
                for (size_t k=0;k<s.size();++k) if (s[k] <= 0) s[k] = 1;
                state_shapes.push_back(s);
                size_t nelems = 1;
                for (auto d : s) nelems *= (size_t)d;
                state_storage.emplace_back(vector<float>(nelems, 0.0f));
            }
 // Prepare head session input/output names (keep allocated names alive)
            Ort::AllocatedStringPtr head_in_alloc = head_sess.GetInputNameAllocated(0, allocator);
            string head_in_name(head_in_alloc.get());
            Ort::AllocatedStringPtr head_out_alloc = head_sess.GetOutputNameAllocated(0, allocator);
            string head_out_name(head_out_alloc.get());

            // streaming loop: each t step supply x_t and states -> get h_out and new states
            int t_dim = T;
            int f_dim = F;
            vector<float> pooled_sum;
            pooled_sum.reserve(1024);
            int count = 0;

            // Timing
            double enc_total_ms = 0.0;
            int enc_calls = 0;
            double head_ms = 0.0;
Ort::MemoryInfo meminfo_default = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            // For each time step
            for (int t=0;t<t_dim;++t) {
                // xbuf: copy frame
                vector<float> xbuf(f_dim);
                memcpy(xbuf.data(), &data[(size_t)t * (size_t)f_dim], sizeof(float) * f_dim);

                // Build inputs: first x_t, then states
                vector<const char*> in_names_c;
                vector<Ort::Value> in_tensors;
                in_names_c.reserve(1 + state_storage.size());
                in_tensors.reserve(1 + state_storage.size());
  // input 0
                in_names_c.push_back(enc_in_names[0].c_str());
                vector<int64_t> xshape = {1, (int64_t)f_dim};
                Ort::Value x_tensor = Ort::Value::CreateTensor<float>(meminfo_default, xbuf.data(), xbuf.size(), xshape.data(), xshape.size());
                in_tensors.push_back(std::move(x_tensor));

                // states
                for (size_t si=0; si<state_storage.size(); ++si) {
                    in_names_c.push_back(enc_in_names[1 + si].c_str());
                    auto &sbuf = state_storage[si];
                    auto &sshape = state_shapes[si];
                    Ort::Value vs = Ort::Value::CreateTensor<float>(meminfo_default, sbuf.data(), sbuf.size(), sshape.data(), sshape.size());
                    in_tensors.push_back(std::move(vs));
                }

                // run encoder
                auto t0 = hrclock::now();
                vector<const char*> out_names_c;
                out_names_c.reserve(enc_out_names.size());
                for (auto &s : enc_out_names) out_names_c.push_back(s.c_str());
                auto outputs = enc_sess.Run(Ort::RunOptions{nullptr}, in_names_c.data(), in_tensors.data(), in_tensors.size(), out_names_c.data(), out_names_c.size());
                auto t1 = hrclock::now();
                enc_calls++;
                enc_total_ms += chrono::duration<double, std::milli>(t1 - t0).count();
 // outputs: outs[0] = h_out (1,d), outs[1...] = new states
                if (outputs.size() < 1) throw runtime_error("Encoder returned no outputs");
                float* hptr = outputs[0].GetTensorMutableData<float>();
                auto hinfo = outputs[0].GetTensorTypeAndShapeInfo();
                auto hshape = hinfo.GetShape(); // expect [1, d]
                int d_dim = (int)hshape.back();
                if (pooled_sum.empty()) pooled_sum.assign(d_dim, 0.0f);
                for (int k=0;k<d_dim;++k) pooled_sum[k] += hptr[k];
                count++;
// read new states
                size_t nstate_out = outputs.size() - 1;
                size_t upto = min(nstate_out, state_storage.size());
                for (size_t si=0; si<upto; ++si) {
                    float* sptr = outputs[1+si].GetTensorMutableData<float>();
                    auto sinfo = outputs[1+si].GetTensorTypeAndShapeInfo();
                    auto sshape = sinfo.GetShape();
                    size_t ne = 1;
                    for (auto x : sshape) ne *= (size_t)x;
                    state_storage[si].assign(sptr, sptr + ne);
                    state_shapes[si] = sshape;
                }
            } // end for t
 // compute pooled mean
            for (auto &v : pooled_sum) v /= max(1, count);

            // prepare head input (1, d)
            vector<int64_t> head_shape = {1, (int64_t)pooled_sum.size()};
            auto t0h = hrclock::now();
            Ort::Value head_in = Ort::Value::CreateTensor<float>(meminfo_default, pooled_sum.data(), pooled_sum.size(), head_shape.data(), head_shape.size());

            const char* head_in_name_c = head_in_name.c_str();
            const char* head_out_name_c = head_out_name.c_str();
            vector<const char*> head_in_names = { head_in_name_c };
            vector<const char*> head_out_names = { head_out_name_c };
            auto head_outs = head_sess.Run(Ort::RunOptions{nullptr}, head_in_names.data(), &head_in, 1, head_out_names.data(), 1);
            auto t1h = hrclock::now();
            head_ms = chrono::duration<double, std::milli>(t1h - t0h).count();
  // parse head output logits
            assert(head_outs.size() >= 1);
            float* outptr = head_outs[0].GetTensorMutableData<float>();
            auto info = head_outs[0].GetTensorTypeAndShapeInfo();
            auto outshape = info.GetShape();
            int nclasses = (int)outshape.back();
            vector<float> logits(outptr, outptr + nclasses);
            // softmax
            float m = *max_element(logits.begin(), logits.end());
            float ssum = 0.0f;
            for (float &v : logits) { v = std::exp(v - m); ssum += v; }
            for (float &v : logits) v /= ssum;
            int best = max_element(logits.begin(), logits.end()) - logits.begin();
cout << "[stream] Top1: " << best << " prob=" << logits[best] << "\n";
            cout << "[timing] encoder total ms: " << enc_total_ms << "  calls: " << enc_calls << "  avg ms/frame: " << (enc_calls? enc_total_ms/enc_calls : 0.0) << "\n";
            cout << "[timing] head ms: " << head_ms << "\n";
            cout << "[timing] total ms (encoder+head): " << (enc_total_ms + head_ms) << "\n";
            return 0;
        }
    } catch (const std::exception &ex) {
        cerr << "ERROR: " << ex.what() << endl;
        return 1;
    }
}


//./host_test_cpp_stream tshm_stream_encoder_step.onnx tshm_stream_head.onnx /workspace/speech_command/left/0a5636ca_nohash_1.bin

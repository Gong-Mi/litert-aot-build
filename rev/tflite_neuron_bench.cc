// tflite_neuron_bench.cc - reverse-engineered NPU access on rothko (MT6989)
// Uses device libtflite_mtk.so (MTK-custom TFLite with Neuron delegate built in).
// The Neuron delegate (TfLiteNeuronDelegateCreate) compiles ANY TFLite model
// on-device via the native NeuroPilot 7.2.4 stack - no litert version checks.
//
// Build (bionic, system linker):
//   aarch64-linux-android-clang++ tflite_neuron_bench.cc \
//     -I tf28_src/tensorflow-2.8.0 -I hdrs \
//     -L/data/local/tmp -l:libtflite_mtk.so -llog \
//     -Wl,-rpath,/data/local/tmp -o tflite_neuron_bench
// Run (root):
//   su -c 'LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/system/lib64 \
//     ./tflite_neuron_bench model.tflite [npu|cpu] [runs]'

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/kernels/register.h"

// MTK Neuron delegate (exported by libtflite_mtk.so)
typedef struct TfLiteNeuronDelegateOptions {
  int verbosity;
  int _pad[7];
} TfLiteNeuronDelegateOptions;

extern "C" {
TfLiteDelegate* TfLiteNeuronDelegateCreate(const TfLiteNeuronDelegateOptions* options);
void TfLiteNeuronDelegateDelete(TfLiteDelegate* delegate);
TfLiteNeuronDelegateOptions TfLiteNeuronDelegateOptionsDefault(void);
}

using tflite::InterpreterBuilder;
using tflite::ops::builtin::BuiltinOpResolver;

static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
  const char* model_path = argc > 1 ? argv[1] : "/data/local/tmp/selfie.tflite";
  bool use_npu = argc > 2 ? atoi(argv[2]) : 1;
  int num_runs = argc > 3 ? atoi(argv[3]) : 20;

  printf("=== TFLite(MTK) Neuron bench on rothko ===\n");
  printf("model: %s\n", model_path);
  printf("mode: %s\n", use_npu ? "NPU (Neuron delegate)" : "CPU (XNNPACK)");

  // Load model
  auto model = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!model) { printf("FAIL: model load\n"); return 1; }
  printf("[ok] model loaded\n");

  BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (builder(&interpreter) != kTfLiteOk || !interpreter) {
    printf("FAIL: interpreter build\n");
    return 1;
  }

  if (use_npu) {
    TfLiteNeuronDelegateOptions opts = TfLiteNeuronDelegateOptionsDefault();
    TfLiteDelegate* del = TfLiteNeuronDelegateCreate(&opts);
    if (!del) { printf("FAIL: neuron delegate create\n"); return 1; }
    printf("[ok] neuron delegate created\n");
    if (interpreter->ModifyGraphWithDelegate(del) != kTfLiteOk) {
      printf("FAIL: ModifyGraphWithDelegate\n");
      return 1;
    }
    printf("[ok] delegate applied\n");
    // delegate is owned by interpreter now? MTK docs: caller owns. Keep ref.
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    printf("FAIL: allocate tensors\n");
    return 1;
  }
  printf("[ok] tensors allocated\n");

  // Fill input with random data
  int num_inputs = interpreter->inputs().size();
  for (int i = 0; i < num_inputs; i++) {
    TfLiteTensor* t = interpreter->input_tensor(i);
    if (!t || !t->data.raw) continue;
    size_t bytes = t->bytes;
    // deterministic pseudo-random
    for (size_t b = 0; b < bytes; b++) {
      t->data.raw[b] = (char)((b * 31 + i * 7) % 251);
    }
  }

  // Warmup
  if (interpreter->Invoke() != kTfLiteOk) {
    printf("FAIL: warmup invoke\n");
    return 1;
  }
  printf("[ok] warmup done\n");

  // Benchmark
  double t0 = now_ms();
  for (int i = 0; i < num_runs; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      printf("FAIL: invoke %d\n", i);
      return 1;
    }
  }
  double total = now_ms() - t0;

  printf("\n=== RESULTS ===\n");
  printf("mode: %s\n", use_npu ? "NPU" : "CPU");
  printf("avg inference: %.2f ms (%d runs)\n", total / num_runs, num_runs);
  printf("total: %.2f ms\n", total);
  return 0;
}

#include "mlp_cuda_kernels.cuh"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <chrono>

#define DEFAULT_EPOCHS 50
#define DEFAULT_BATCH_SIZE 32
#define DEFAULT_LEARNING_RATE 0.05f
#define DEFAULT_PATIENCE 8
#define EARLY_STOPPING_MIN_DELTA 0.0001f

typedef struct { int count; int features; float* x; float* y; } Dataset;
typedef struct { float* w1; float* b1; float* w2; float* b2; float* w3; float* b3; } HostWeights;
typedef struct { float* a1; float* a2; float* out; float* d1; float* d2; float* d3; float* gw1; float* gb1; float* gw2; float* gb2; float* gw3; float* gb3; } CpuBuffers;

static double now_ms(void) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}
static void* checked_malloc(size_t bytes) { void* p = malloc(bytes); if (!p) fprintf(stderr, "Memoria insuficiente.\n"); return p; }

static int load_dataset(const char* path, Dataset* data) {
    FILE* file = fopen(path, "rb"); char magic[8]; unsigned int count, features;
    memset(data, 0, sizeof(*data));
    if (!file || fread(magic, 1, 8, file) != 8 || memcmp(magic, "YAWNMLP", 7) != 0 || fread(&count, 4, 1, file) != 1 || fread(&features, 4, 1, file) != 1 || features != CUDA_INPUT_DIM) {
        if (file) fclose(file); return 0;
    }
    data->count = (int)count; data->features = (int)features;
    data->y = (float*)checked_malloc((size_t)count * sizeof(float));
    data->x = (float*)checked_malloc((size_t)count * features * sizeof(float));
    if (!data->x || !data->y) { fclose(file); free(data->x); free(data->y); return 0; }
    {
        int* labels = (int*)checked_malloc((size_t)count * sizeof(int));
        if (!labels || fread(labels, sizeof(int), count, file) != count || fread(data->x, sizeof(float), (size_t)count * features, file) != (size_t)count * features) {
            free(labels); fclose(file); free(data->x); free(data->y); return 0;
        }
        for (unsigned int i = 0; i < count; ++i) data->y[i] = (float)labels[i];
        free(labels);
    }
    fclose(file); return 1;
}

static void free_dataset(Dataset* data) { free(data->x); free(data->y); memset(data, 0, sizeof(*data)); }
static int shuffle_dataset(const Dataset* source, Dataset* shuffled) {
    unsigned int state = 42u;
    int* indices = (int*)checked_malloc((size_t)source->count * sizeof(int));
    memset(shuffled, 0, sizeof(*shuffled));
    if (!indices) return 0;
    shuffled->count = source->count; shuffled->features = source->features;
    shuffled->x = (float*)checked_malloc((size_t)source->count * source->features * sizeof(float));
    shuffled->y = (float*)checked_malloc((size_t)source->count * sizeof(float));
    if (!shuffled->x || !shuffled->y) { free(indices); free_dataset(shuffled); return 0; }
    for (int i = 0; i < source->count; ++i) indices[i] = i;
    for (int i = source->count - 1; i > 0; --i) { state = state * 1664525u + 1013904223u; int j = (int)(state % (unsigned int)(i + 1)); int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp; }
    for (int i = 0; i < source->count; ++i) { int source_index = indices[i]; memcpy(shuffled->x + (size_t)i * source->features, source->x + (size_t)source_index * source->features, (size_t)source->features * sizeof(float)); shuffled->y[i] = source->y[source_index]; }
    free(indices); return 1;
}

static int allocate_weights(HostWeights* w) {
    memset(w, 0, sizeof(*w));
    w->w1 = (float*)checked_malloc((size_t)CUDA_INPUT_DIM * CUDA_HIDDEN1 * sizeof(float)); w->b1 = (float*)checked_malloc(CUDA_HIDDEN1 * sizeof(float));
    w->w2 = (float*)checked_malloc((size_t)CUDA_HIDDEN1 * CUDA_HIDDEN2 * sizeof(float)); w->b2 = (float*)checked_malloc(CUDA_HIDDEN2 * sizeof(float));
    w->w3 = (float*)checked_malloc(CUDA_HIDDEN2 * sizeof(float)); w->b3 = (float*)checked_malloc(sizeof(float));
    return w->w1 && w->b1 && w->w2 && w->b2 && w->w3 && w->b3;
}
static void free_weights(HostWeights* w) { free(w->w1); free(w->b1); free(w->w2); free(w->b2); free(w->w3); free(w->b3); memset(w, 0, sizeof(*w)); }
static int write_weights(const char* path, const HostWeights* w) {
    FILE* file = fopen(path, "wb"); char magic[8] = "CUDAWTS"; uint32_t dims[3] = {CUDA_INPUT_DIM, CUDA_HIDDEN1, CUDA_HIDDEN2};
    if (!file) return 0;
    fwrite(magic, 1, 8, file); fwrite(dims, sizeof(uint32_t), 3, file);
    fwrite(w->w1, sizeof(float), (size_t)CUDA_INPUT_DIM * CUDA_HIDDEN1, file); fwrite(w->b1, sizeof(float), CUDA_HIDDEN1, file);
    fwrite(w->w2, sizeof(float), (size_t)CUDA_HIDDEN1 * CUDA_HIDDEN2, file); fwrite(w->b2, sizeof(float), CUDA_HIDDEN2, file);
    fwrite(w->w3, sizeof(float), CUDA_HIDDEN2, file); fwrite(w->b3, sizeof(float), 1, file); fclose(file); return 1;
}
static void initialize_weights(HostWeights* w) {
    size_t sizes[] = {(size_t)CUDA_INPUT_DIM * CUDA_HIDDEN1, CUDA_HIDDEN1, (size_t)CUDA_HIDDEN1 * CUDA_HIDDEN2, CUDA_HIDDEN2, CUDA_HIDDEN2, 1};
    float* arrays[] = {w->w1, w->b1, w->w2, w->b2, w->w3, w->b3};
    for (int a = 0; a < 6; ++a) for (size_t i = 0; i < sizes[a]; ++i) arrays[a][i] = a % 2 ? 0.0f : (((int)((i * 37u + (unsigned)a * 19u) % 1000u) / 1000.0f) - 0.5f) * 0.04f;
}
static void copy_weights(HostWeights* dst, const HostWeights* src) {
    memcpy(dst->w1, src->w1, (size_t)CUDA_INPUT_DIM * CUDA_HIDDEN1 * sizeof(float)); memcpy(dst->b1, src->b1, CUDA_HIDDEN1 * sizeof(float));
    memcpy(dst->w2, src->w2, (size_t)CUDA_HIDDEN1 * CUDA_HIDDEN2 * sizeof(float)); memcpy(dst->b2, src->b2, CUDA_HIDDEN2 * sizeof(float));
    memcpy(dst->w3, src->w3, CUDA_HIDDEN2 * sizeof(float)); memcpy(dst->b3, src->b3, sizeof(float));
}

static int allocate_cpu_buffers(CpuBuffers* b, int batch) {
    memset(b, 0, sizeof(*b));
    b->a1=(float*)checked_malloc((size_t)batch*CUDA_HIDDEN1*sizeof(float)); b->a2=(float*)checked_malloc((size_t)batch*CUDA_HIDDEN2*sizeof(float)); b->out=(float*)checked_malloc((size_t)batch*sizeof(float));
    b->d1=(float*)checked_malloc((size_t)batch*CUDA_HIDDEN1*sizeof(float)); b->d2=(float*)checked_malloc((size_t)batch*CUDA_HIDDEN2*sizeof(float)); b->d3=(float*)checked_malloc((size_t)batch*sizeof(float));
    b->gw1=(float*)checked_malloc((size_t)CUDA_INPUT_DIM*CUDA_HIDDEN1*sizeof(float)); b->gb1=(float*)checked_malloc(CUDA_HIDDEN1*sizeof(float));
    b->gw2=(float*)checked_malloc((size_t)CUDA_HIDDEN1*CUDA_HIDDEN2*sizeof(float)); b->gb2=(float*)checked_malloc(CUDA_HIDDEN2*sizeof(float)); b->gw3=(float*)checked_malloc(CUDA_HIDDEN2*sizeof(float)); b->gb3=(float*)checked_malloc(sizeof(float));
    return b->a1&&b->a2&&b->out&&b->d1&&b->d2&&b->d3&&b->gw1&&b->gb1&&b->gw2&&b->gb2&&b->gw3&&b->gb3;
}
static void free_cpu_buffers(CpuBuffers* b) { free(b->a1);free(b->a2);free(b->out);free(b->d1);free(b->d2);free(b->d3);free(b->gw1);free(b->gb1);free(b->gw2);free(b->gb2);free(b->gw3);free(b->gb3); }

static void forward_cpu(const float* x, int batch, const HostWeights* w, CpuBuffers* b) {
    for(int r=0;r<batch;++r){ for(int j=0;j<CUDA_HIDDEN1;++j){float s=w->b1[j];for(int i=0;i<CUDA_INPUT_DIM;++i)s+=x[r*CUDA_INPUT_DIM+i]*w->w1[i*CUDA_HIDDEN1+j];b->a1[r*CUDA_HIDDEN1+j]=s>0?s:0;} }
    for(int r=0;r<batch;++r){ for(int j=0;j<CUDA_HIDDEN2;++j){float s=w->b2[j];for(int i=0;i<CUDA_HIDDEN1;++i)s+=b->a1[r*CUDA_HIDDEN1+i]*w->w2[i*CUDA_HIDDEN2+j];b->a2[r*CUDA_HIDDEN2+j]=s>0?s:0;} }
    for(int r=0;r<batch;++r){float s=w->b3[0];for(int i=0;i<CUDA_HIDDEN2;++i)s+=b->a2[r*CUDA_HIDDEN2+i]*w->w3[i];b->out[r]=1.0f/(1.0f+expf(-s));}
}
static void train_cpu_batch(const float* x,const float* y,int batch,HostWeights* w,CpuBuffers* b,float lr,float* loss,float* correct) {
    forward_cpu(x,batch,w,b); *loss=0;*correct=0;
    for(int r=0;r<batch;++r){float p=fminf(fmaxf(b->out[r],1e-7f),1-1e-7f);*loss+=-(y[r]*logf(p)+(1-y[r])*logf(1-p));*correct+=((p>=0.5f)==(y[r]>=0.5f));b->d3[r]=b->out[r]-y[r];} *loss /= (float)batch;
    for(int r=0;r<batch;++r)for(int j=0;j<CUDA_HIDDEN2;++j){float s=b->d3[r]*w->w3[j];b->d2[r*CUDA_HIDDEN2+j]=b->a2[r*CUDA_HIDDEN2+j]>0?s:0;}
    for(int r=0;r<batch;++r)for(int j=0;j<CUDA_HIDDEN1;++j){float s=0;for(int k=0;k<CUDA_HIDDEN2;++k)s+=b->d2[r*CUDA_HIDDEN2+k]*w->w2[j*CUDA_HIDDEN2+k];b->d1[r*CUDA_HIDDEN1+j]=b->a1[r*CUDA_HIDDEN1+j]>0?s:0;}
    for(int i=0;i<CUDA_HIDDEN2;++i){float s=0;for(int r=0;r<batch;++r)s+=b->a2[r*CUDA_HIDDEN2+i]*b->d3[r];b->gw3[i]=s/batch;} *b->gb3=0;for(int r=0;r<batch;++r)*b->gb3+=b->d3[r]/batch;
    for(int i=0;i<CUDA_HIDDEN1;++i)for(int j=0;j<CUDA_HIDDEN2;++j){float s=0;for(int r=0;r<batch;++r)s+=b->a1[r*CUDA_HIDDEN1+i]*b->d2[r*CUDA_HIDDEN2+j];b->gw2[i*CUDA_HIDDEN2+j]=s/batch;} for(int j=0;j<CUDA_HIDDEN2;++j){float s=0;for(int r=0;r<batch;++r)s+=b->d2[r*CUDA_HIDDEN2+j];b->gb2[j]=s/batch;}
    for(int i=0;i<CUDA_INPUT_DIM;++i)for(int j=0;j<CUDA_HIDDEN1;++j){float s=0;for(int r=0;r<batch;++r)s+=x[r*CUDA_INPUT_DIM+i]*b->d1[r*CUDA_HIDDEN1+j];b->gw1[i*CUDA_HIDDEN1+j]=s/batch;} for(int j=0;j<CUDA_HIDDEN1;++j){float s=0;for(int r=0;r<batch;++r)s+=b->d1[r*CUDA_HIDDEN1+j];b->gb1[j]=s/batch;}
    for(int i=0;i<CUDA_INPUT_DIM*CUDA_HIDDEN1;++i)w->w1[i]-=lr*b->gw1[i];for(int i=0;i<CUDA_HIDDEN1;++i)w->b1[i]-=lr*b->gb1[i];for(int i=0;i<CUDA_HIDDEN1*CUDA_HIDDEN2;++i)w->w2[i]-=lr*b->gw2[i];for(int i=0;i<CUDA_HIDDEN2;++i)w->b2[i]-=lr*b->gb2[i];for(int i=0;i<CUDA_HIDDEN2;++i)w->w3[i]-=lr*b->gw3[i];w->b3[0]-=lr*(*b->gb3);
}

static void evaluate_cpu(const Dataset* test,const HostWeights* w,CpuBuffers* b,float* loss,float* accuracy){float total=0,correct=0;for(int i=0;i<test->count;++i){forward_cpu(test->x+i*CUDA_INPUT_DIM,1,w,b);float p=fminf(fmaxf(b->out[0],1e-7f),1-1e-7f);total-=test->y[i]*logf(p)+(1-test->y[i])*logf(1-p);correct+=((p>=.5f)==(test->y[i]>=.5f));}*loss=total/test->count;*accuracy=correct/test->count;}

static int allocate_device(float** p,size_t count){return cudaMalloc((void**)p,count*sizeof(float))==cudaSuccess;}
static void free_device(CudaMlpWeights*w,CudaMlpActivations*a,CudaMlpGradients*g,float*x,float*y){cudaFree(x);cudaFree(y);cudaFree(w->w1);cudaFree(w->b1);cudaFree(w->w2);cudaFree(w->b2);cudaFree(w->w3);cudaFree(w->b3);cudaFree(a->a1);cudaFree(a->a2);cudaFree(a->output);cudaFree(g->d1);cudaFree(g->d2);cudaFree(g->d3);cudaFree(g->losses);cudaFree(g->gw1);cudaFree(g->gb1);cudaFree(g->gw2);cudaFree(g->gb2);cudaFree(g->gw3);cudaFree(g->gb3);}

static void copy_host_to_device(CudaMlpWeights* dst, const HostWeights* src) {
    cudaMemcpy(dst->w1,src->w1,(size_t)CUDA_INPUT_DIM*CUDA_HIDDEN1*sizeof(float),cudaMemcpyHostToDevice); cudaMemcpy(dst->b1,src->b1,CUDA_HIDDEN1*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dst->w2,src->w2,(size_t)CUDA_HIDDEN1*CUDA_HIDDEN2*sizeof(float),cudaMemcpyHostToDevice); cudaMemcpy(dst->b2,src->b2,CUDA_HIDDEN2*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dst->w3,src->w3,CUDA_HIDDEN2*sizeof(float),cudaMemcpyHostToDevice); cudaMemcpy(dst->b3,src->b3,sizeof(float),cudaMemcpyHostToDevice);
}

static void copy_device_to_host(HostWeights* dst, const CudaMlpWeights* src) {
    cudaMemcpy(dst->w1,src->w1,(size_t)CUDA_INPUT_DIM*CUDA_HIDDEN1*sizeof(float),cudaMemcpyDeviceToHost); cudaMemcpy(dst->b1,src->b1,CUDA_HIDDEN1*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->w2,src->w2,(size_t)CUDA_HIDDEN1*CUDA_HIDDEN2*sizeof(float),cudaMemcpyDeviceToHost); cudaMemcpy(dst->b2,src->b2,CUDA_HIDDEN2*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(dst->w3,src->w3,CUDA_HIDDEN2*sizeof(float),cudaMemcpyDeviceToHost); cudaMemcpy(dst->b3,src->b3,sizeof(float),cudaMemcpyDeviceToHost);
}

int main(int argc, char** argv) {
    int epochs = argc > 7 ? atoi(argv[7]) : DEFAULT_EPOCHS;
    int batch_size = argc > 8 ? atoi(argv[8]) : DEFAULT_BATCH_SIZE;
    float lr = argc > 9 ? (float)atof(argv[9]) : DEFAULT_LEARNING_RATE;
    int patience = argc > 10 ? atoi(argv[10]) : DEFAULT_PATIENCE;
    Dataset train, validation, test, shuffled;
    HostWeights initial, cpu, cpu_best, gpu_current, gpu_best;
    CpuBuffers buffers;
    CudaMlpWeights gw = {0}; CudaMlpActivations ga = {0}; CudaMlpGradients gg = {0};
    float *dx = 0, *dy = 0, *host_out, *host_losses;
    float *cpu_loss, *cpu_acc, *cpu_val_loss, *cpu_val_acc, *gpu_loss, *gpu_acc, *gpu_val_loss, *gpu_val_acc;
    int gpu_epochs = 0, best_gpu_epoch = 0, best_cpu_epoch = 0, gpu_wait = 0;
    float best_gpu_val_loss = INFINITY, best_cpu_val_loss = INFINITY;
    double gpu_time = 0.0, cpu_time = 0.0;

    if (argc < 7 || epochs <= 0 || batch_size <= 0 || patience <= 0 ||
        !load_dataset(argv[1], &train) || !load_dataset(argv[2], &validation) || !load_dataset(argv[3], &test) ||
        !allocate_weights(&initial) || !allocate_weights(&cpu) || !allocate_weights(&cpu_best) ||
        !allocate_weights(&gpu_current) || !allocate_weights(&gpu_best) || !allocate_cpu_buffers(&buffers, batch_size)) {
        fprintf(stderr, "Uso: mlp_cuda train.bin validation.bin test.bin reporte.md reporte.json pesos.bin [epocas] [batch] [learning_rate] [patience]\n");
        return 1;
    }
    if (!shuffle_dataset(&train, &shuffled)) return 1;
    initialize_weights(&initial);
    cpu_loss = (float*)calloc(epochs, sizeof(float)); cpu_acc = (float*)calloc(epochs, sizeof(float));
    cpu_val_loss = (float*)calloc(epochs, sizeof(float)); cpu_val_acc = (float*)calloc(epochs, sizeof(float));
    gpu_loss = (float*)calloc(epochs, sizeof(float)); gpu_acc = (float*)calloc(epochs, sizeof(float));
    gpu_val_loss = (float*)calloc(epochs, sizeof(float)); gpu_val_acc = (float*)calloc(epochs, sizeof(float));
    host_out = (float*)malloc(batch_size * sizeof(float)); host_losses = (float*)malloc(batch_size * sizeof(float));
    if (!cpu_loss || !cpu_acc || !cpu_val_loss || !cpu_val_acc || !gpu_loss || !gpu_acc || !gpu_val_loss || !gpu_val_acc || !host_out || !host_losses ||
        !allocate_device(&dx, (size_t)batch_size * CUDA_INPUT_DIM) || !allocate_device(&dy, batch_size) ||
        !allocate_device(&gw.w1, (size_t)CUDA_INPUT_DIM * CUDA_HIDDEN1) || !allocate_device(&gw.b1, CUDA_HIDDEN1) ||
        !allocate_device(&gw.w2, (size_t)CUDA_HIDDEN1 * CUDA_HIDDEN2) || !allocate_device(&gw.b2, CUDA_HIDDEN2) ||
        !allocate_device(&gw.w3, CUDA_HIDDEN2) || !allocate_device(&gw.b3, 1) ||
        !allocate_device(&ga.a1, (size_t)batch_size * CUDA_HIDDEN1) || !allocate_device(&ga.a2, (size_t)batch_size * CUDA_HIDDEN2) || !allocate_device(&ga.output, batch_size) ||
        !allocate_device(&gg.d1, (size_t)batch_size * CUDA_HIDDEN1) || !allocate_device(&gg.d2, (size_t)batch_size * CUDA_HIDDEN2) || !allocate_device(&gg.d3, batch_size) || !allocate_device(&gg.losses, batch_size) ||
        !allocate_device(&gg.gw1, (size_t)CUDA_INPUT_DIM * CUDA_HIDDEN1) || !allocate_device(&gg.gb1, CUDA_HIDDEN1) || !allocate_device(&gg.gw2, (size_t)CUDA_HIDDEN1 * CUDA_HIDDEN2) || !allocate_device(&gg.gb2, CUDA_HIDDEN2) || !allocate_device(&gg.gw3, CUDA_HIDDEN2) || !allocate_device(&gg.gb3, 1)) {
        fprintf(stderr, "No se pudo reservar memoria.\n"); return 1;
    }

    /* CUDA usa val_loss para elegir y conservar la mejor version del modelo. */
    copy_host_to_device(&gw, &initial);
    for (int e = 0; e < epochs; ++e) {
        float total_loss = 0, total_acc = 0; double start = now_ms();
        for (int s = 0; s < shuffled.count; s += batch_size) {
            int n = s + batch_size <= shuffled.count ? batch_size : shuffled.count - s;
            cudaMemcpy(dx, shuffled.x + (size_t)s * CUDA_INPUT_DIM, (size_t)n * CUDA_INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(dy, shuffled.y + s, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
            cuda_forward_pass(dx, &gw, &ga, n); binary_cross_entropy_kernel<<<(n + 255) / 256, 256>>>(ga.output, dy, gg.losses, n);
            cudaMemcpy(host_out, ga.output, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost); cudaMemcpy(host_losses, gg.losses, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
            for (int r = 0; r < n; ++r) { total_loss += host_losses[r]; total_acc += ((host_out[r] >= .5f) == (shuffled.y[s + r] >= .5f)); }
            cuda_backward_update(dx, dy, &gw, &ga, &gg, n, lr);
        }
        cudaDeviceSynchronize(); gpu_time += now_ms() - start;
        gpu_loss[e] = total_loss / shuffled.count; gpu_acc[e] = total_acc / shuffled.count;
        copy_device_to_host(&gpu_current, &gw); evaluate_cpu(&validation, &gpu_current, &buffers, &gpu_val_loss[e], &gpu_val_acc[e]); gpu_epochs = e + 1;
        printf("CUDA epoca %d/%d - loss: %.4f - accuracy: %.4f - val_loss: %.4f - val_accuracy: %.4f\n", gpu_epochs, epochs, gpu_loss[e], gpu_acc[e], gpu_val_loss[e], gpu_val_acc[e]);
        if (gpu_val_loss[e] < best_gpu_val_loss - EARLY_STOPPING_MIN_DELTA) { best_gpu_val_loss = gpu_val_loss[e]; copy_weights(&gpu_best, &gpu_current); best_gpu_epoch = gpu_epochs; gpu_wait = 0; }
        else if (++gpu_wait >= patience) { printf("CUDA detenido por EarlyStopping en la epoca %d.\n", gpu_epochs); break; }
    }

    /* CPU ejecuta las mismas epocas efectivas para que el speedup mida el mismo trabajo. */
    copy_weights(&cpu, &initial);
    for (int e = 0; e < gpu_epochs; ++e) {
        float total_loss = 0, total_acc = 0; double start = now_ms();
        for (int s = 0; s < shuffled.count; s += batch_size) {
            int n = s + batch_size <= shuffled.count ? batch_size : shuffled.count - s; float loss, acc;
            train_cpu_batch(shuffled.x + (size_t)s * CUDA_INPUT_DIM, shuffled.y + s, n, &cpu, &buffers, lr, &loss, &acc);
            total_loss += loss * n; total_acc += acc;
        }
        cpu_time += now_ms() - start; cpu_loss[e] = total_loss / shuffled.count; cpu_acc[e] = total_acc / shuffled.count;
        evaluate_cpu(&validation, &cpu, &buffers, &cpu_val_loss[e], &cpu_val_acc[e]);
        if (cpu_val_loss[e] < best_cpu_val_loss - EARLY_STOPPING_MIN_DELTA) { best_cpu_val_loss = cpu_val_loss[e]; copy_weights(&cpu_best, &cpu); best_cpu_epoch = e + 1; }
    }
    copy_weights(&cpu, &cpu_best);
    float cpu_test_loss, cpu_test_acc, gpu_test_loss, gpu_test_acc;
    evaluate_cpu(&test, &cpu, &buffers, &cpu_test_loss, &cpu_test_acc); evaluate_cpu(&test, &gpu_best, &buffers, &gpu_test_loss, &gpu_test_acc);

    /* El speedup es tiempo CPU de entrenamiento dividido por tiempo CUDA de entrenamiento. */
    FILE* report = fopen(argv[4], "w"); FILE* json = fopen(argv[5], "w");
    if (!report || !json) { fprintf(stderr, "No se pudo escribir resultados.\n"); return 1; }
    fprintf(report, "# Entrenamiento MLP: CPU vs CUDA\n\nArquitectura: `4096 -> 64 -> 32 -> 1`. Maximo: **%d** epocas; efectivas: **%d**; batch: **%d**.\n\n## EarlyStopping\n\nCUDA supervisa `val_loss`, con paciencia **%d**, `min_delta=%.4f` y `restore_best_weights=True`. La mejor epoca CUDA fue la **%d** (val_loss **%.4f**). CPU repite las mismas epocas efectivas para medir exactamente el mismo trabajo y conserva su mejor peso de validacion (epoca %d).\n\nEl tiempo mide solo los bucles de entrenamiento: forward, BCE, backpropagation, SGD, transferencias por batch y sincronizacion CUDA. Excluye preprocesamiento, compilacion, validacion y evaluacion final.\n\n| Plataforma | Tiempo (ms) | Loss prueba | Accuracy prueba | Speedup |\n|---|---:|---:|---:|---:|\n| CPU serial | %.3f | %.4f | %.4f | 1.000x |\n| CUDA GPU | %.3f | %.4f | %.4f | %.3fx |\n\n| Epoca | Loss CPU | Accuracy CPU | Val loss CPU | Val accuracy CPU | Loss CUDA | Accuracy CUDA | Val loss CUDA | Val accuracy CUDA |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n", epochs, gpu_epochs, batch_size, patience, EARLY_STOPPING_MIN_DELTA, best_gpu_epoch, best_gpu_val_loss, best_cpu_epoch, cpu_time, cpu_test_loss, cpu_test_acc, gpu_time, gpu_test_loss, gpu_test_acc, cpu_time / gpu_time);
    for (int e = 0; e < gpu_epochs; ++e) fprintf(report, "| %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |\n", e + 1, cpu_loss[e], cpu_acc[e], cpu_val_loss[e], cpu_val_acc[e], gpu_loss[e], gpu_acc[e], gpu_val_loss[e], gpu_val_acc[e]);
    fclose(report);
    fprintf(json, "{\n  \"measurement_scope\": \"full_training_loop_all_effective_epochs_all_batches\",\n  \"epochs_requested\": %d,\n  \"epochs\": %d,\n  \"batch_size\": %d,\n  \"early_stopping\": {\"monitor\": \"val_loss\", \"patience\": %d, \"min_delta\": %.6f, \"restore_best_weights\": true, \"best_epoch\": %d, \"best_val_loss\": %.6f},\n", epochs, gpu_epochs, batch_size, patience, EARLY_STOPPING_MIN_DELTA, best_gpu_epoch, best_gpu_val_loss);
    fprintf(json, "  \"cpu\": {\"time_ms\": %.6f, \"test_loss\": %.6f, \"test_accuracy\": %.6f, \"best_epoch\": %d, \"loss\": [", cpu_time, cpu_test_loss, cpu_test_acc, best_cpu_epoch);
    for (int e = 0; e < gpu_epochs; ++e) fprintf(json, "%s%.6f", e ? ", " : "", cpu_loss[e]); fprintf(json, "], \"accuracy\": ["); for (int e = 0; e < gpu_epochs; ++e) fprintf(json, "%s%.6f", e ? ", " : "", cpu_acc[e]); fprintf(json, "], \"val_loss\": ["); for (int e = 0; e < gpu_epochs; ++e) fprintf(json, "%s%.6f", e ? ", " : "", cpu_val_loss[e]); fprintf(json, "], \"val_accuracy\": ["); for (int e = 0; e < gpu_epochs; ++e) fprintf(json, "%s%.6f", e ? ", " : "", cpu_val_acc[e]); fprintf(json, "]},\n");
    fprintf(json, "  \"cuda\": {\"time_ms\": %.6f, \"test_loss\": %.6f, \"test_accuracy\": %.6f, \"best_epoch\": %d, \"loss\": [", gpu_time, gpu_test_loss, gpu_test_acc, best_gpu_epoch);
    for (int e = 0; e < gpu_epochs; ++e) fprintf(json, "%s%.6f", e ? ", " : "", gpu_loss[e]); fprintf(json, "], \"accuracy\": ["); for (int e = 0; e < gpu_epochs; ++e) fprintf(json, "%s%.6f", e ? ", " : "", gpu_acc[e]); fprintf(json, "], \"val_loss\": ["); for (int e = 0; e < gpu_epochs; ++e) fprintf(json, "%s%.6f", e ? ", " : "", gpu_val_loss[e]); fprintf(json, "], \"val_accuracy\": ["); for (int e = 0; e < gpu_epochs; ++e) fprintf(json, "%s%.6f", e ? ", " : "", gpu_val_acc[e]); fprintf(json, "]}\n}\n"); fclose(json);
    if (!write_weights(argv[6], &gpu_best)) { fprintf(stderr, "No se pudieron guardar los pesos CUDA.\n"); return 1; }
    printf("Entrenamiento CPU/CUDA finalizado con EarlyStopping y pesos CUDA de la mejor validacion exportados.\n");
    free_device(&gw, &ga, &gg, dx, dy); free(host_out); free(host_losses); free(cpu_loss); free(cpu_acc); free(cpu_val_loss); free(cpu_val_acc); free(gpu_loss); free(gpu_acc); free(gpu_val_loss); free(gpu_val_acc);
    free_cpu_buffers(&buffers); free_weights(&initial); free_weights(&cpu); free_weights(&cpu_best); free_weights(&gpu_current); free_weights(&gpu_best); free_dataset(&shuffled); free_dataset(&train); free_dataset(&validation); free_dataset(&test); return 0;
}

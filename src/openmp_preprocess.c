#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define INPUT_WIDTH 128
#define INPUT_HEIGHT 128
#define OUTPUT_WIDTH 64
#define OUTPUT_HEIGHT 64
#define FEATURES (OUTPUT_WIDTH * OUTPUT_HEIGHT)

typedef struct { int count; int width; int height; int* labels; uint8_t* pixels; } RawDataset;

static int clamp_value(int value, int low, int high) { return value < low ? low : (value > high ? high : value); }

static int load_raw(const char* path, RawDataset* data) {
    FILE* file = fopen(path, "rb"); char magic[8]; uint32_t count, width, height; size_t pixels_size;
    memset(data, 0, sizeof(*data));
    if (!file || fread(magic, 1, 8, file) != 8 || memcmp(magic, "OMPRAW64", 8) != 0 || fread(&count, 4, 1, file) != 1 || fread(&width, 4, 1, file) != 1 || fread(&height, 4, 1, file) != 1 || width != INPUT_WIDTH || height != INPUT_HEIGHT) { if (file) fclose(file); return 0; }
    pixels_size = (size_t)count * width * height * 3u;
    data->labels = (int*)malloc((size_t)count * sizeof(int)); data->pixels = (uint8_t*)malloc(pixels_size);
    if (!data->labels || !data->pixels || fread(data->labels, sizeof(int), count, file) != count || fread(data->pixels, 1, pixels_size, file) != pixels_size) { fclose(file); free(data->labels); free(data->pixels); return 0; }
    fclose(file); data->count = (int)count; data->width = (int)width; data->height = (int)height; return 1;
}

static void free_raw(RawDataset* data) { free(data->labels); free(data->pixels); memset(data, 0, sizeof(*data)); }

static void preprocess_one(const uint8_t* pixels, float* output) {
    /* Convierte RGB a gris, suaviza con Gaussiano y genera 4096 valores normalizados. */
    float gray[INPUT_WIDTH * INPUT_HEIGHT]; float blur[INPUT_WIDTH * INPUT_HEIGHT];
    for (int i = 0; i < INPUT_WIDTH * INPUT_HEIGHT; ++i) gray[i] = 0.299f * pixels[i * 3] + 0.587f * pixels[i * 3 + 1] + 0.114f * pixels[i * 3 + 2];
    for (int y = 0; y < INPUT_HEIGHT; ++y) for (int x = 0; x < INPUT_WIDTH; ++x) {
        float sum = 0.0f; static const float k[3][3]={{1,2,1},{2,4,2},{1,2,1}};
        for (int ky=-1; ky<=1; ++ky) for (int kx=-1; kx<=1; ++kx) sum += gray[clamp_value(y+ky,0,INPUT_HEIGHT-1)*INPUT_WIDTH+clamp_value(x+kx,0,INPUT_WIDTH-1)] * k[ky+1][kx+1];
        blur[y*INPUT_WIDTH+x] = sum / 16.0f;
    }
    for (int y=0; y<OUTPUT_HEIGHT; ++y) for (int x=0; x<OUTPUT_WIDTH; ++x) {
        float source_x=(float)x*(INPUT_WIDTH-1)/(OUTPUT_WIDTH-1), source_y=(float)y*(INPUT_HEIGHT-1)/(OUTPUT_HEIGHT-1); int x0=(int)source_x,y0=(int)source_y,x1=clamp_value(x0+1,0,INPUT_WIDTH-1),y1=clamp_value(y0+1,0,INPUT_HEIGHT-1); float wx=source_x-x0,wy=source_y-y0;
        float top=blur[y0*INPUT_WIDTH+x0]*(1-wx)+blur[y0*INPUT_WIDTH+x1]*wx, bottom=blur[y1*INPUT_WIDTH+x0]*(1-wx)+blur[y1*INPUT_WIDTH+x1]*wx;
        output[y*OUTPUT_WIDTH+x]=(top*(1-wy)+bottom*wy)/255.0f;
    }
}

static double process_dataset(const RawDataset* data, float* output, int threads) {
    double start=omp_get_wtime();
    if (threads > 0) {
        omp_set_num_threads(threads);
        /* Cada iteracion procesa una imagen independiente: OpenMP puede repartirlas entre hilos. */
        #pragma omp parallel for schedule(static)
        for (int i=0; i<data->count; ++i) preprocess_one(data->pixels+(size_t)i*INPUT_WIDTH*INPUT_HEIGHT*3u, output+(size_t)i*FEATURES);
    } else {
        for (int i=0; i<data->count; ++i) preprocess_one(data->pixels+(size_t)i*INPUT_WIDTH*INPUT_HEIGHT*3u, output+(size_t)i*FEATURES);
    }
    return (omp_get_wtime()-start)*1000.0;
}

static int write_processed(const char* path, const RawDataset* data, const float* features) {
    FILE* file=fopen(path,"wb"); char magic[8]="YAWNMLP"; if(!file) return 0;
    fwrite(magic,1,8,file); {uint32_t count=(uint32_t)data->count, dims=FEATURES; fwrite(&count,4,1,file); fwrite(&dims,4,1,file);} fwrite(data->labels,sizeof(int),data->count,file); fwrite(features,sizeof(float),(size_t)data->count*FEATURES,file); fclose(file); return 1;
}

int main(int argc, char** argv) {
    if (argc != 9) { fprintf(stderr,"Uso: openmp_preprocess raw_train raw_validation raw_test out_train out_validation out_test reporte_md reporte_json\n"); return 1; }
    RawDataset train,validation,test;
    float *train_serial,*validation_serial,*test_serial,*train_omp,*validation_omp,*test_omp;
    double serial_ms,omp_ms; FILE* md,*json; int images;
    if(!load_raw(argv[1],&train)||!load_raw(argv[2],&validation)||!load_raw(argv[3],&test)){fprintf(stderr,"No se pudieron leer los datos exportados.\n");return 1;}
    train_serial=(float*)malloc((size_t)train.count*FEATURES*sizeof(float)); validation_serial=(float*)malloc((size_t)validation.count*FEATURES*sizeof(float)); test_serial=(float*)malloc((size_t)test.count*FEATURES*sizeof(float));
    train_omp=(float*)malloc((size_t)train.count*FEATURES*sizeof(float)); validation_omp=(float*)malloc((size_t)validation.count*FEATURES*sizeof(float)); test_omp=(float*)malloc((size_t)test.count*FEATURES*sizeof(float));
    if(!train_serial||!validation_serial||!test_serial||!train_omp||!validation_omp||!test_omp)return 1;
    /* El speedup compara una pasada completa serial frente a la misma pasada con ocho hilos. */
    serial_ms=process_dataset(&train,train_serial,0)+process_dataset(&validation,validation_serial,0)+process_dataset(&test,test_serial,0);
    omp_ms=process_dataset(&train,train_omp,8)+process_dataset(&validation,validation_omp,8)+process_dataset(&test,test_omp,8);
    if(!write_processed(argv[4],&train,train_omp)||!write_processed(argv[5],&validation,validation_omp)||!write_processed(argv[6],&test,test_omp))return 1;
    md=fopen(argv[7],"w");json=fopen(argv[8],"w");if(!md||!json)return 1;
    images=train.count+validation.count+test.count;
    fprintf(md,"# Preprocesamiento: serial vs OpenMP\n\nAlcance: una pasada completa sobre todas las imagenes de entrenamiento, validacion y prueba en memoria. Incluye escala de grises, Gaussiano, resize, normalizacion y flatten. Excluye lectura/escritura de archivos, recorte facial, inferencia y entrenamiento.\n\n| Modo | Hilos | Imagenes | Tiempo (ms) | Speedup |\n|---|---:|---:|---:|---:|\n| Serial | 1 | %d | %.3f | 1.000x |\n| OpenMP | 8 | %d | %.3f | %.3fx |\n",images,serial_ms,images,omp_ms,serial_ms/omp_ms);
    fprintf(json,"{\n  \"measurement_scope\": \"full_preprocessing_train_validation_and_test_in_memory\",\n  \"serial_ms\": %.6f,\n  \"openmp_ms\": %.6f,\n  \"threads\": 8,\n  \"images\": %d,\n  \"speedup\": %.6f\n}\n",serial_ms,omp_ms,images,serial_ms/omp_ms);
    fclose(md);fclose(json);free(train_serial);free(validation_serial);free(test_serial);free(train_omp);free(validation_omp);free(test_omp);free_raw(&train);free_raw(&validation);free_raw(&test);return 0;
}

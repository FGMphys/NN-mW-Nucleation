#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

unsigned int OMPThreadsNum=8;
void setOMPthreads(unsigned int num){OMPThreadsNum=num;};

void computeSmoothMax(double *ds,int num_ds,double *alpha,int num_alpha_radiale,
                      double *sds, int dimbat, int N, int* numneigh,double* type_emb2b,
                      int nt, int* type_map)
{
  int par,b,i;
  int numdes=num_ds;
  // costruiamo i descrittori
#pragma omp parallel for num_threads(OMPThreadsNum)
for (b=0;b<dimbat;b++){
    for (par=0;par<N;par++){
        int num=numneigh[b*N*(numdes+1)+par*(numdes+1)];
        int actual=b*N*numdes+par*numdes;
        //Proiezione su base e derivazione del singolo termine proiettato per la particella par nel frame b
        int j;
        for (j=0;j<num;j++){
            int neighj=numneigh[b*N*(numdes+1)+par*(numdes+1)+1+j];
            int ch_type=type_map[neighj];
        for (i=0;i<num_alpha_radiale;i++)
            {
              double buffer1=ds[actual+j]*exp(alpha[i]*ds[actual+j])*type_emb2b[ch_type];
              sds[b*num_alpha_radiale*N+par*num_alpha_radiale+i]+=buffer1;
             }
             }
         }
     }
}







using namespace tensorflow;

REGISTER_OP("ComputeSortProj")
    .Input("sortdes: double")
    .Input("numpar: int32")
    .Input("numnn: int32")
    .Input("dimbatch: int32")
    .Input("numneigh: int32")
    .Input("alpharad: double")
    .Input("nalpha_r: int32")
    .Input("type_emb2b: double")
    .Input("nt: int32")
    .Input("type_map: int32")
    .Output("sortprojdes: double");





class ComputeSortProjOp : public OpKernel {
 public:
  explicit ComputeSortProjOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor0 = context->input(0);
    const Tensor& input_tensor1 = context->input(1);
    const Tensor& input_tensor2 = context->input(2);
    const Tensor& input_tensor3 = context->input(3);
    const Tensor& input_tensor4 = context->input(4);
    const Tensor& input_tensor5 = context->input(5);
    const Tensor& input_tensor6 = context->input(6);
    //emebedding work
    const Tensor& input_tensor7 = context->input(7);
    const Tensor& input_tensor8 = context->input(8);
    const Tensor& input_tensor9 = context->input(9);

    auto input0 = input_tensor0.flat<double>();
    auto input1 = input_tensor1.flat<int>();
    auto input2 = input_tensor2.flat<int>();
    auto input3 = input_tensor3.flat<int>();
    auto input4 = input_tensor4.flat<int>();
    auto input5 = input_tensor5.flat<double>();
    auto input6 = input_tensor6.flat<int>();

    //emebedding work
    auto input7 = input_tensor7.flat<double>();
    auto input8 = input_tensor8.flat<int>();
    auto input9 = input_tensor9.flat<int>();


    //Copio i tensori input in nuovi array per elaborarli
    int dimbat=input3(0);
    int numdes=input2(0);
    int N=input1(0);
    int dimdes=dimbat*numdes*N;
    int num_alpha_radiale=input6(0);

    //Work embedding
    int nt=input8(0);
    double* type_emb2b;
    type_emb2b=(double*)malloc(sizeof(double)*nt);
    for (int i = 0; i < nt; i++) {
    type_emb2b[i]=input7(i);
    }
    int* type_map;
    type_map=(int*)malloc(sizeof(int)*N);
    for (int i = 0; i < N; i++) {
    type_map[i]=input9(i);
    }
    /////////////////////////////////

    //Leggo il descrittore di rete
    double* radiale;
    radiale=(double*)malloc(sizeof(double)*dimdes);
    for (int i = 0; i < dimdes; i++) {
    radiale[i]=input0(i);
    }
    //Leggo il numero di vicini nel tempo
    int* numneigh;
    numneigh=(int*)malloc(sizeof(int)*dimbat*N*(numdes+1));
    for (int k=0;k<dimbat*N*(numdes+1);k++){
        numneigh[k]=input4(k);
    }
    //Alloco l'array di output
    double* smoothradiale=(double*)malloc(num_alpha_radiale*N*dimbat*sizeof(double));
    for (int k=0;k<num_alpha_radiale*N*dimbat;k++){
        smoothradiale[k]=0.;
    }
    //Definisco i termini dell'espansione da considerare
    double* alpha_radiale;
    alpha_radiale=(double*)malloc(num_alpha_radiale*sizeof(double));
    for (int i=0;i<num_alpha_radiale;i++){
        alpha_radiale[i]=input5(i);
        }
    //Calcolo della proiezione su base
    computeSmoothMax(radiale,numdes,alpha_radiale,num_alpha_radiale,
                     smoothradiale,dimbat,N,numneigh,type_emb2b,nt,type_map);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (num_alpha_radiale);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();

    //Copio element per elemento l'array di output nel tensore di output
    for (int i = 0; i < (dimbat*N*num_alpha_radiale); i++) {
    output_flat(i)=smoothradiale[i];
    }
    free(smoothradiale);
    free(alpha_radiale);
    free(radiale);
    free(numneigh);
    free(type_map);
    free(type_emb2b);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSortProj").Device(DEVICE_CPU), ComputeSortProjOp);

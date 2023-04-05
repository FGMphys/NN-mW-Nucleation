///Implementazione del gradiente di una funzione scalare L(SD), funzione dei SD(alpha).


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <omp.h>
#include <thread>

unsigned int OMPThreadsNum=8;
void setOMPthreads(unsigned int num){OMPThreadsNum=num;};


void compute_2bodyalphagrad(double *ds,int numdes,double *alpha,int nalpha_r,
                            double *nextgrad, int dimbat,
                            int N, int* numneigh, double *prevgrad,int nt,
                            double* type_emb2b,int* type_map,
                            double* nextgrad_emb2b)
{
  int par,b,i;
  double prevgradel;
  // costruiamo i descrittori
#pragma omp parallel for num_threads(OMPThreadsNum)
for (b=0;b<dimbat;b++){

    for (par=0;par<N;par++){
        int num=numneigh[b*N*(numdes+1)+par*(numdes+1)];
        int actual=b*N*numdes+par*numdes;
        int j;
        for (j=0;j<num;j++)
        {
          int neighj=numneigh[b*N*(numdes+1)+par*(numdes+1)+1+j];
          int cht=type_map[neighj];
          double typew=type_emb2b[cht];
          for (i=0;i<nalpha_r;i++){
              double ds_el=ds[actual+j];
              prevgradel=prevgrad[b*nalpha_r*N+par*nalpha_r+i];
              nextgrad[i]+=ds_el*ds_el*exp(alpha[i]*ds_el)*typew*prevgradel;
              nextgrad_emb2b[cht]+=ds_el*exp(alpha[i]*ds_el)*prevgradel;
             }
          }
           }
         }
       }












using namespace tensorflow;

REGISTER_OP("ComputeTwoBodyParGrad")
    .Input("prev_grad: double")
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
    .Output("nextgrad_alpha2b: double")
    .Output("nextgrad_emb2b: double");





class ComputeTwoBodyParGradOp : public OpKernel {
 public:
  explicit ComputeTwoBodyParGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor0 = context->input(0);
    const Tensor& input_tensor1 = context->input(1);
    const Tensor& input_tensor2 = context->input(2);
    const Tensor& input_tensor3 = context->input(3);
    const Tensor& input_tensor4 = context->input(4);
    const Tensor& input_tensor5 = context->input(5);
    const Tensor& input_tensor6 = context->input(6);
    const Tensor& input_tensor7 = context->input(7);

    //working embeddign 2b
    const Tensor& input_tensor8 = context->input(8);
    const Tensor& input_tensor9 = context->input(9);
    const Tensor& input_tensor10 = context->input(10);



    auto input0 = input_tensor0.flat<double>();
    auto input1 = input_tensor1.flat<double>();
    auto input2 = input_tensor2.flat<int>();
    auto input3 = input_tensor3.flat<int>();
    auto input4 = input_tensor4.flat<int>();
    auto input5 = input_tensor5.flat<int>();
    auto input6 = input_tensor6.flat<double>();
    auto input7 = input_tensor7.flat<int>();

    //working embedding 2b
    auto input8 = input_tensor8.flat<double>();
    auto input9 = input_tensor9.flat<int>();
    auto input10 = input_tensor10.flat<int>();
   // auto input8 = input_tensor8.flat<int>();
    //Copio i tensori input in nuovi array per elaborarli
    int dimbat=input4(0);
    int numdes=input3(0);
    int N=input2(0);
    int dimdes=dimbat*numdes*N;
    int nalpha_r=input7(0);
   // int dim_decoder=input8(0)
    //Leggo il descrittore di rete
    double* radiale;
    radiale=(double*)malloc(sizeof(double)*dimdes);
    for (int i = 0; i < dimdes; i++) {
    radiale[i]=input1(i);
    }
    //Leggo il numero di vicini nel tempo
    int* numneigh;
    numneigh=(int*)malloc(sizeof(int)*dimbat*N*(numdes+1));
    for (int k=0;k<dimbat*N*(numdes+1);k++){
        numneigh[k]=input5(k);
    }
    //Definisco i termini dell'espansione da considerare
    double* alpha_radiale;
    alpha_radiale=(double*)malloc(nalpha_r*sizeof(double));
    for (int i=0;i<nalpha_r;i++){
        alpha_radiale[i]=input6(i);//(double*)malloc(nalpha_r*sizeof(double));
        }
    //Leggo il gradiente del layer precedente
    double* prev_grad;
    prev_grad=(double*)malloc(dimbat*N*nalpha_r*sizeof(double));
    for (int i=0;i<dimbat*N*nalpha_r;i++)  {
        prev_grad[i]=input0(i);
    }

    //working embedding 2b
    int nt=input9(0);
    double* type_emb2b=(double*)calloc(nt,sizeof(double));
    for (int i=0;i<nt;i++)  {
        type_emb2b[i]=input8(i);
    }
    int* type_map=(int*)calloc(N,sizeof(int));
    for (int i=0;i<N;i++)  {
        type_map[i]=input10(i);
    }




    //Alloco gli array di output e inzializzo dato che sommo ricorsivamente
    double* nextgrad=(double*)calloc(nalpha_r,sizeof(double));

    double* nextgrad_emb2b=(double*)calloc(nt,sizeof(double));


    //Calcolo della proiezione su base
    compute_2bodyalphagrad(radiale,numdes,alpha_radiale,nalpha_r,
                           nextgrad,dimbat,N,numneigh,
                           prev_grad,nt,type_emb2b,type_map,nextgrad_emb2b);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nalpha_r);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();

    //Create output tensor for backprob of embedding 2b params
    Tensor* output_tensor2 = NULL;
    TensorShape grad_net_shape2 ;
    grad_net_shape2.AddDim (nt);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_net_shape2,
                                                     &output_tensor2));
    auto output_flat2 = output_tensor2->flat<double>();
    //Copio element per elemento l'array di output nel tensore di output
    for (int i = 0; i < (nalpha_r); i++) {
    output_flat(i)=nextgrad[i];
    }
    for (int i = 0; i < (nt); i++) {
    output_flat2(i)=nextgrad_emb2b[i];
    }

    free(nextgrad);
    free(prev_grad);
    free(alpha_radiale);
    free(radiale);
    free(numneigh);
    free(nextgrad_emb2b);
    free(type_map);
    free(type_emb2b);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeTwoBodyParGrad").Device(DEVICE_CPU), ComputeTwoBodyParGradOp);

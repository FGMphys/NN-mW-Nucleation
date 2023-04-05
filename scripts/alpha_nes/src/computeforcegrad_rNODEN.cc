#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

#include <omp.h>
#include <thread>

unsigned int OMPThreadsNum=8;
void setOMPthreads(unsigned int num){OMPThreadsNum=num;};


void back_prop_grad_forceop(double* prevgrad, double* ds,int numdes,double* alpha,
                            int num_alpha_radiale,double* intderiv,int* nnindex,
                            int dimbat,int N,double* nextgrad,double* nextgrad2,
                            double* NG,double* type_emb2b, int nt,
                            int* type_map,double* nextgrad_emb2b)
                            {


//info for cycle: batch-->particles-->directions--->descriptors
  for (int b=0; b<dimbat; b++){
    #pragma omp parallel for num_threads(OMPThreadsNum)
     for (int par = 0; par < N; par++) {
         int actual=b*N*numdes+par*numdes;
         int num_neigh=nnindex[b*(N*(numdes+1))+(numdes+1)*par];

         for (int j=0;j<num_neigh;j++){
              int neighj=nnindex[b*(N*(numdes+1))+(numdes+1)*par+1+j];
              int ch_type=type_map[neighj];
              double chpar=type_emb2b[ch_type];

              double ds_el=ds[actual+j];
              for (int a =0; a<3; a++){
                double prevgrad_el=prevgrad[b*(N*3)+par*3+a];
                double prevgrad_neigh=prevgrad[b*(N*3)+neighj*3+a];
                double common = 0.5*intderiv[b*N*3*numdes+numdes*3*par+a*numdes+j];

                 for (int i=0;i<num_alpha_radiale;i++){
                      double alpha_el=alpha[i];
                      double supp1=exp(alpha_el*ds_el);
                      double sds_deriv=supp1*(1.+alpha_el*ds_el);
                      double buff_alpha=chpar*supp1*ds_el*(2.+alpha_el*ds_el);

                      double  NGel=NG[b*N*num_alpha_radiale+par*num_alpha_radiale+i];
                      int index_sup=b*(N*num_alpha_radiale)+par*num_alpha_radiale+i;

                      nextgrad[index_sup] -= prevgrad_el*common*chpar*sds_deriv;
                      nextgrad[index_sup] += prevgrad_neigh*common*chpar*sds_deriv;

                      nextgrad2[i]-=prevgrad_el*NGel*buff_alpha*common;
                      nextgrad2[i]+=prevgrad_neigh*NGel*buff_alpha*common;

                      nextgrad_emb2b[ch_type]-=prevgrad_el*NGel*sds_deriv*common;
                      nextgrad_emb2b[ch_type]+=prevgrad_neigh*NGel*sds_deriv*common;



               }
        }




               }
           }
         }
}























using namespace tensorflow;

REGISTER_OP("ComputeForceRadialGrad")
    .Input("grad: double")
    .Input("netderiv: double")
    .Input("intderiv: double")
    .Input("nnindex: int32")
    .Input("numpar: int32")
    .Input("numnn: int32")
    .Input("dimbatch: int32")
    .Input("sortdes: double")
    .Input("nalpha_r: int32")
    .Input("alpharad: double")
    .Input("type_emb2b: double")
    .Input("nt_couple: int32")
    .Input("type_map: int32")
    .Output("gradnet: double")
    .Output("nextgrad2: double")
    .Output("grad_emb2b: double");




class ComputeForceRadialGradOp : public OpKernel {
 public:
  explicit ComputeForceRadialGradOp(OpKernelConstruction* context) : OpKernel(context) {}

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
    const Tensor& input_tensor8 = context->input(8);
    const Tensor& input_tensor9 = context->input(9);

    const Tensor& input_tensor10 = context->input(10);
    const Tensor& input_tensor11 = context->input(11);
    const Tensor& input_tensor12 = context->input(12);



    auto input0 = input_tensor0.flat<double>();
    auto input1 =  input_tensor1.flat<double>();
    auto input2 = input_tensor2.flat<double>();
    auto input3 = input_tensor3.flat<int>();
    auto input4 = input_tensor4.flat<int>();
    auto input5 = input_tensor5.flat<int>();
    auto input6 = input_tensor6.flat<int>();
    auto input7 = input_tensor7.flat<double>();
    auto input8 = input_tensor8.flat<int>();
    auto input9 = input_tensor9.flat<double>();

    auto input10 = input_tensor10.flat<double>();
    auto input11 = input_tensor11.flat<int>();
    auto input12 = input_tensor12.flat<int>();

    //Copio i tensori input in nuovi array per elaborarli
    int dimbat=input6(0);
    int numdes=input5(0);
    int num_alpha_radiale=input8(0);
    int max_input=num_alpha_radiale;
    int N=input4(0);
    int dimnet=dimbat*num_alpha_radiale*N;
    int dimdes=dimbat*numdes*N;

    //working embedding
    int nt=input11(0);
    double* type_emb2b;
    type_emb2b=(double*)malloc(sizeof(double)*nt);
    for (int i = 0; i < nt; i++) {
    type_emb2b[i]=input10(i);
    }
    int* type_map;
    type_map=(int*)malloc(sizeof(int)*N);
    for (int i = 0; i < N; i++) {
    type_map[i]=input12(i);
    }

   //Leggo il gradiente esterno
    double* lastgrad; //last back-prpagated gradient
    lastgrad=(double*)malloc(sizeof(double)*dimbat*N*3);
    for (int i = 0; i < dimbat*N*3; i++) {
    lastgrad[i]=input0(i);
    }
    double* NG; //net derivative
    NG=(double*)malloc(sizeof(double)*dimnet);
    for (int i = 0; i < dimnet; i++) {
    NG[i]=input1(i);
    }
    //Leggo la derivata dei descrittori
    double* int_deriv;
    int_deriv=(double*)malloc(sizeof(double)*dimdes*3);
    for (int i = 0; i < (dimdes*3); i++) {
    int_deriv[i]=input2(i);
    }
    //Leggo la mappa di interazione
    int* nnindex;
    nnindex=(int*)malloc(sizeof(int)*dimbat*(numdes+1)*N);
    for (int i = 0; i < dimbat*(numdes+1)*N; i++) {
    nnindex[i]=input3(i);
    }
    //Leggo il descrittore di rete
    double* radiale;
    radiale=(double*)malloc(sizeof(double)*dimdes);
    for (int i = 0; i < dimdes; i++) {
    radiale[i]=input7(i);
    }
    //Definisco i termini dell'espansione da considerare
    double alpha_radiale_modmax=input8(0);
    double* alpha_radiale=(double*)malloc(num_alpha_radiale*sizeof(double));
    for (int i=0; i< num_alpha_radiale; i++){
        alpha_radiale[i]=input9(i);
        }
    //Alloco l'array di output
    double* newgrad_backprop; //gradient after back-propagation through force layer
    newgrad_backprop=(double*)calloc(dimnet,sizeof(double));
    double* nextgrad2;
    nextgrad2=(double*)calloc(num_alpha_radiale,sizeof(double));
    double* nextgrad_emb2b;
    nextgrad_emb2b=(double*)calloc(nt,sizeof(double));
    //Calcolo dei descrittori
    back_prop_grad_forceop(lastgrad,radiale,numdes,alpha_radiale,num_alpha_radiale,
                           int_deriv,nnindex,dimbat,
                           N,newgrad_backprop,nextgrad2,NG,type_emb2b,
                           nt,type_map,nextgrad_emb2b);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (1);
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (num_alpha_radiale);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();

    Tensor* output_tensor2 = NULL;
    TensorShape grad_net_shape2 ;
    grad_net_shape2.AddDim (num_alpha_radiale);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_net_shape2,
                                                     &output_tensor2));
    auto output_flat2 = output_tensor2->flat<double>();

    Tensor* output_tensor3 = NULL;
    TensorShape grad_net_shape3 ;
    grad_net_shape3.AddDim (nt);
    OP_REQUIRES_OK(context, context->allocate_output(2, grad_net_shape3,
                                                     &output_tensor3));
    auto output_flat3 = output_tensor3->flat<double>();

    //Copio element per elemento l'array di output nel tensore di output
    for (int i = 0; i < (num_alpha_radiale*N*dimbat); i++) {
    output_flat(i)=newgrad_backprop[i];
    }
    for (int i = 0; i < (num_alpha_radiale); i++) {
    output_flat2(i)=nextgrad2[i];
    }
    for (int i = 0; i < nt; i++) {
    output_flat3(i)=nextgrad_emb2b[i];
    }

    free(type_map);
    free(type_emb2b);
    free(nextgrad_emb2b);
    free(newgrad_backprop);
    free(nextgrad2);
    free(lastgrad);
    free(int_deriv);
    free(nnindex);
    free(NG);
    free(alpha_radiale);
    free(radiale);

  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadialGrad").Device(DEVICE_CPU), ComputeForceRadialGradOp);

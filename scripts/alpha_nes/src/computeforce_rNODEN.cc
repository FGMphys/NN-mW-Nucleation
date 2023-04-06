#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"



void computeSmoothMaxForce(double *ds,int num_ds,double *alpha,int num_alpha_radiale,
                          double* grad,double* intderiv, int* nnindex, int dimbat,
                          int N,double* force,double* type_emb2b,int nt,
                          int* type_map)
{
  int par2,b,i;
  int numdes=num_ds;
  // costruiamo i descrittori
for (b=0;b<dimbat;b++){

    for (par2=0;par2<N;par2++){
        int actual=b*N*numdes+par2*numdes;
        int num_neigh=nnindex[b*(N*(numdes+1))+(numdes+1)*par2];
        for (int j=0;j<num_neigh;j++)
        {

          double des_r=ds[actual+j];
          int neighj=nnindex[b*(N*(numdes+1))+(numdes+1)*par2+1+j];
          int ch_type=type_map[neighj];
          double chpar=type_emb2b[ch_type];
          for (int a =0; a<3; a++){
              double intder_r=intderiv[b*N*3*numdes+numdes*3*par2+a*numdes+j];
              for (int i=0; i<num_alpha_radiale;i++){
                  double sds_deriv=chpar*exp(alpha[i]*des_r)*(1.+alpha[i]*des_r);
                  double prevgrad=grad[b*N*num_alpha_radiale+num_alpha_radiale*par2+i];
                  double temp = 0.5*sds_deriv*intder_r;
                  force[b*(N*3)+3*par2+a] -= prevgrad*temp;
                  force[b*(N*3)+3*neighj+a] += prevgrad*temp;
                 }
               }

            }
}
}
}

using namespace tensorflow;

REGISTER_OP("ComputeForceRadial")
    .Input("netderiv: double")
    .Input("intderiv: double")
    .Input("nnindex: int32")
    .Input("numpar2: int32")
    .Input("numnn: int32")
    .Input("dimbatch: int32")
    .Input("sortdes: double")
    .Input("nalpha_r: int32")
    .Input("alpharad: double")
    .Input("type_emb2b: double")
    .Input("nt: int32")
    .Input("type_map: int32")
    .Output("force: double");

class ComputeForceRadialOp : public OpKernel {
 public:
  explicit ComputeForceRadialOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);
    const Tensor& input_tensor3 = context->input(2);
    const Tensor& input_tensor4 = context->input(3);
    const Tensor& input_tensor5 = context->input(4);
    const Tensor& input_tensor6 = context->input(5);
    const Tensor& input_tensor7 = context->input(6);
    const Tensor& input_tensor8 = context->input(7);
    const Tensor& input_tensor9 = context->input(8);

    const Tensor& input_tensor10 = context->input(9);
    const Tensor& input_tensor11 = context->input(10);
    const Tensor& input_tensor12 = context->input(11);


    auto input = input_tensor1.flat<double>();
    auto input2 = input_tensor2.flat<double>();
    auto input3 = input_tensor3.flat<int>();
    auto input4 = input_tensor4.flat<int>();
    auto input5 = input_tensor5.flat<int>();
    auto input6 = input_tensor6.flat<int>();
    auto input7 = input_tensor7.flat<double>();
    auto input8 = input_tensor8.flat<int>();
    auto input9 = input_tensor9.flat<double>();

    //working emb2b force2b
    auto input10 = input_tensor10.flat<double>();
    auto input11 = input_tensor11.flat<int>();
    auto input12 = input_tensor12.flat<int>();

    //Copio i tensori input in nuovi array per elaborarli
    int dimbat=input6(0);
    int numdes=input5(0);
    int N=input4(0);
    int dimdes=dimbat*numdes*N;
    int num_alpha_radiale=input8(0);
    int dimnetderiv=num_alpha_radiale*dimbat*N;
    //working type_emb_2b
    //Work embedding
    int nt=input11(0);
    double* type_emb2b;
    type_emb2b=(double*)malloc(nt*sizeof(double));
    for (int i = 0; i < nt; i++) {
    type_emb2b[i]=input10(i);
    }
    int* type_map;
    type_map=(int*)malloc(sizeof(int)*N);
    for (int i = 0; i < N; i++) {
    type_map[i]=input12(i);
    }
    //Reading net derivative
    double* net_deriv;
    net_deriv=(double*)malloc(sizeof(double)*dimnetderiv);
    for (int i = 0; i < dimnetderiv; i++) {
    net_deriv[i]=input(i);
    }
    //Leggo il descrittore di rete
    double* radiale;
    radiale=(double*)malloc(sizeof(double)*dimdes);
    for (int i = 0; i < dimdes; i++) {
    radiale[i]=input7(i);
    }
    //Leggo la derivata del descrittore
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
    //Alloco l'array di output
    double* force;
    force=(double*)calloc(sizeof(double),dimbat*N*3);
    //Definisco i termini dell'espansione da considerare
    //double alpha_radiale_modmax=input8(0);
    double* alpha_radiale=(double*)malloc(num_alpha_radiale*sizeof(double));
    for (int i=0; i< num_alpha_radiale; i++){
        alpha_radiale[i]=input9(i);
        }

    //Calcolo delle forze
    computeSmoothMaxForce(radiale,numdes,alpha_radiale,num_alpha_radiale,
                          net_deriv,int_deriv,nnindex,dimbat,N,force,
                          type_emb2b,nt,type_map);


    // Create an output tensor
    Tensor* forces2b = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (3);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &forces2b));
    auto forces2bflat = forces2b->flat<double>();
    //Copio element per elemento l'array di output nel tensore di output
    for (int i = 0; i < (dimbat*N*3); i++) {
    forces2bflat(i)=force[i];
    }


    free(force);
    free(net_deriv);
    free(int_deriv);
    free(nnindex);
    free(alpha_radiale);
    free(radiale);
    free(type_map);
    free(type_emb2b);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceRadial").Device(DEVICE_CPU), ComputeForceRadialOp);

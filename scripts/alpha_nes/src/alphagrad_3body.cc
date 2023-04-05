#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <omp.h>
#include <thread>


unsigned int OMPThreadsNum=8;
void setOMPthreads(unsigned int num){OMPThreadsNum=num;};

void compute_nextgrad(double *ds, double *ds3bod, int numdsrad, int numds3bod,
                      double *prevgrad, int dimbat, int N, int *neigh3body,
                      double *smooth_a, int nsmooth_a, double *nextgrad,
                      double** type_emb3b,int nt,int* type_map,
                      double** nextgrad_emb3b)
{
  // costruiamo i descrittori
#pragma omp parallel for num_threads(OMPThreadsNum)
for (int b=0;b<dimbat;b++){

    for (int par=0;par<N;par++){
        int nne3bod=neigh3body[b*N*(numds3bod*2+1)+par*(numds3bod*2+1)];
        int actual=b*N*numdsrad+par*numdsrad;
        int actual_ang=b*N*numds3bod+par*numds3bod;
        int aux2=0;
        //Costruzione della proiezione sia della parte 2body  che 3body....////
       ////////////////////////////////////////////////////////////////////////
        for (int j=0;j<nne3bod-1;j++){
           for (int y=j+1;y<nne3bod;y++){
               int neighj=neigh3body[b*N*(numds3bod*2+1)+par*(numds3bod*2+1)+1+aux2*2];
               int neighy=neigh3body[b*N*(numds3bod*2+1)+par*(numds3bod*2+1)+1+aux2*2+1];

               int j_type=type_map[neighj];
               int y_type=type_map[neighy];
               double chtjy_par=type_emb3b[j_type][y_type];

               double dj=ds[actual+j];
               double dy=ds[actual+y];
               double Tjy=ds3bod[actual_ang+aux2];

               for (int a1=0;a1<nsmooth_a;a1++){
                    double betaval=smooth_a[a1*3+2];
                    double alpha1=smooth_a[a1*3+0];
                    double alpha2=smooth_a[a1*3+1];
                    //NUMERATORE//
                    double prevgradel=prevgrad[b*nsmooth_a*N+par*nsmooth_a+a1];
                     //Derivate rispetto beta gamma delta
                     double a1dja2dy=exp(alpha1*dj+alpha2*dy);
                     double a1dya2dj=exp(alpha1*dy+alpha2*dj);
                     double btjy=exp(betaval*Tjy);
                     //derivate rispetto emb3body par
                     nextgrad_emb3b[j_type][y_type]+=prevgradel*Tjy*btjy*(a1dya2dj+a1dja2dy)/2.;
                     nextgrad_emb3b[y_type][j_type]+=prevgradel*Tjy*btjy*(a1dya2dj+a1dja2dy)/2.;

                     nextgrad[a1*3]+=prevgradel*chtjy_par*(a1dja2dy*dj+a1dya2dj*dy)*btjy*Tjy/2.;
                     nextgrad[a1*3+1]+=prevgradel*chtjy_par*(a1dja2dy*dy+a1dya2dj*dj)*btjy*Tjy/2.;
                     nextgrad[a1*3+2]+=prevgradel*chtjy_par*(a1dja2dy+a1dya2dj)*btjy*Tjy*Tjy/2.;


                   }
                   aux2++;
              }

	    }
	}
    }
  //Riscalo la diagonale eliminando il double counting
  for (int k=0;k<nt;k++){
      nextgrad_emb3b[k][k]*=0.5;
    }
  }






using namespace tensorflow;

REGISTER_OP("ComputeSortProj3bodyGrad")
    .Input("prevgrad: double")
    .Input("threebodydes: double")
    .Input("twobodydes: double")
    .Input("numdes3body: int32")
    .Input("numdes2body: int32")
    .Input("num_neigh3body: int32")
    .Input("numpar: int32")
    .Input("dimbatch: int32")
    .Input("num_neigh2body: int32")
    .Input("alphabeta_a: double")
    .Input("nsmooth_a: int32")
    .Input("type_emb3b: double")
    .Input("nt: int32")
    .Input("type_map: int32")
    .Output("alphagrad3body: double")
    .Output("nextgrad_emb3b: double");





class ComputeSortProj3bodyGradOp : public OpKernel {
 public:
  explicit ComputeSortProj3bodyGradOp(OpKernelConstruction* context) : OpKernel(context) {}

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

    //working embedding 3b
    const Tensor& input_tensor12 = context->input(11);
    const Tensor& input_tensor13 = context->input(12);
    const Tensor& input_tensor14 = context->input(13);


    auto input1 = input_tensor1.flat<double>();
    auto input2 = input_tensor2.flat<double>();
    auto input3 = input_tensor3.flat<double>();
    auto input4 = input_tensor4.flat<int>();
    auto input5 = input_tensor5.flat<int>();
    auto input6 = input_tensor6.flat<int>();

    auto input7 = input_tensor7.flat<int>();
    auto input8 = input_tensor8.flat<int>();
    auto input9 = input_tensor9.flat<int>();
    auto input10 = input_tensor10.flat<double>();
    auto input11 = input_tensor11.flat<int>();

    //working embedding
    auto input12 = input_tensor12.flat<double>();
    auto input13 = input_tensor13.flat<int>();
    auto input14 = input_tensor14.flat<int>();
    //Copio i tensori input in nuovi array per elaborarli
    int dimbat=input8(0);
    int numdes2body=input5(0);
    int numdes3body=input4(0);
    int N=input7(0);
    int dimdes2body=dimbat*numdes2body*N;
    int dimdes3body=dimbat*numdes3body*N;
    int nsmooth_a=input11(0);

    //working embedding
    int nt=input13(0);
    int nt_couple=nt+nt*(nt-1)/2;
    double** type_emb3b;
    type_emb3b=(double**)calloc(nt,sizeof(double*));
    for (int k=0;k<nt;k++){
        type_emb3b[k]=(double*)calloc(nt,sizeof(double));
       }
    int k=0;
    for (int i = 0; i < nt; i++) {
        for (int j = i; j < nt; j++){
            type_emb3b[i][j]=input12(k);
            type_emb3b[j][i]=input12(k);
            k=k+1;
          }
    }
    int* type_map;
    type_map=(int*)calloc(N,sizeof(int));
    for (int i = 0; i < N; i++) {
    type_map[i]=input14(i);
    }

    //Leggo il descrittore due corpi
    double* radiale;
    radiale=(double*)malloc(sizeof(double)*dimdes2body);
    for (int i = 0; i < dimdes2body; i++) {
    radiale[i]=input3(i);
    }
    //Leggo il descrittore 3 corpi
    double* angolare;
    angolare=(double*)malloc(sizeof(double)*dimdes3body);
    for (int i = 0; i < dimdes3body; i++) {
    angolare[i]=input2(i);
    }
    //Leggo il numero di vicini nel tempo
    int* numneigh3body;
    numneigh3body=(int*)malloc(sizeof(int)*dimbat*N*(numdes3body*2+1));
    for (int k=0;k<dimbat*N*(numdes3body*2+1);k++){
        numneigh3body[k]=input6(k);
    }

    // Leggo il gradiente del layer precedente
    double* prevgrad;
    prevgrad=(double*)malloc(sizeof(double)*dimbat*N*nsmooth_a);
    for (int k=0;k<dimbat*N*nsmooth_a;k++){
        prevgrad[k]=input1(k);
    }
    //Alloco gli alpha 3 corpi
    double* smooth_a=(double*)malloc(nsmooth_a*3*sizeof(double));
    for (int i=0; i< nsmooth_a*3; i++){
        smooth_a[i]=input10(i);
        }

    //Alloco l'array di output e inzializzo dato che sommo recorsivamente
    double* nextgrad=(double*)calloc(nsmooth_a*3,sizeof(double));
    double** nextgrad_emb3b=(double**)calloc(nt,sizeof(double*));
    for (int k=0;k<nt;k++){
        nextgrad_emb3b[k]=(double*)calloc(nt,sizeof(double));
       }

    //Calcolo della proiezione su base
    compute_nextgrad(radiale,angolare,numdes2body,numdes3body,prevgrad,dimbat,
                     N,numneigh3body,smooth_a,nsmooth_a,nextgrad,type_emb3b,
                   nt,type_map,nextgrad_emb3b);


    // Create an output tensor for beta gamma delta AFs
    Tensor* output_tensor = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (nsmooth_a);
    grad_net_shape.AddDim (3);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();

    // Create an output for embedding 3b
    Tensor* output_tensor2 = NULL;
    TensorShape grad_net_shape2 ;
    grad_net_shape2.AddDim (nt_couple);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_net_shape2,
                                                     &output_tensor2));
    auto output_flat2 = output_tensor2->flat<double>();

    //Copio element per elemento l'array di output nel tensore di output
    for (int i = 0; i < (nsmooth_a*3); i++) {
    output_flat(i)=nextgrad[i];
    }
    k=0;
    for (int i = 0; i < nt; i++) {
        for (int j = i; j < nt; j++){
             output_flat2(k)=nextgrad_emb3b[i][j];
             k=k+1;
           }
    }


   free(nextgrad);
   free(prevgrad);
   free(smooth_a);
   free(radiale);
   free(angolare);
   free(numneigh3body);
   for (int k=0;k<nt;k++){
         free(nextgrad_emb3b[k]);
    }
   free(nextgrad_emb3b);
   for (int k=0;k<nt;k++){
         free(type_emb3b[k]);
    }
    free(type_emb3b);
    free(type_map);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSortProj3bodyGrad").Device(DEVICE_CPU), ComputeSortProj3bodyGradOp);

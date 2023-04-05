#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


#include <omp.h>
#include <thread>


unsigned int OMPThreadsNum=8;
void setOMPthreads(unsigned int num){OMPThreadsNum=num;};

void computeSmoothMax3body(double *ds, double *ds3bod, int numdsrad,
                           int numds3bod, double *sds,
                           int dimbat, int N, int *neigh3body, double *smooth_a,
                           int nsmooth_a, double** type_emb3b,int nt,
                           int* type_map)
{
  // costruiamo i descrittori
#pragma omp parallel for num_threads(OMPThreadsNum)
for (int b=0;b<dimbat;b++){
    for (int par=0;par<N;par++){
        int nne3bod=neigh3body[b*N*(numds3bod*2+1)+par*(numds3bod*2+1)];
        int actual=b*N*numdsrad+par*numdsrad;
        int actual_ang=b*N*numds3bod+par*numds3bod;
        int aux2=0;
        for (int j=0;j<nne3bod-1;j++){
            for (int y=j+1;y<nne3bod;y++){
                 int neighj=neigh3body[b*N*(numds3bod*2+1)+par*(numds3bod*2+1)+1+aux2*2];
                 int neighy=neigh3body[b*N*(numds3bod*2+1)+par*(numds3bod*2+1)+1+aux2*2+1];

                 int j_type=type_map[neighj];
                 int y_type=type_map[neighy];
                 double chtjy_par=type_emb3b[j_type][y_type];

                 double angulardes=ds3bod[actual_ang+aux2];
               for (int a1=0;a1<nsmooth_a;a1++){

                     double betaval=smooth_a[a1*3+2];
                     double alpha1=smooth_a[a1*3+0];
                     double alpha2=smooth_a[a1*3+1];
                     double softmaxweight=exp(alpha1*ds[actual+j]+alpha2*ds[actual+y]);
                     softmaxweight+=exp(alpha2*ds[actual+j]+alpha1*ds[actual+y]);
                     softmaxweight*=exp(betaval*angulardes);
                     sds[b*nsmooth_a*N+par*nsmooth_a+a1]+=angulardes*softmaxweight*chtjy_par/2.;
                    }
               aux2++;
              }
          }
	     }
    }
}





using namespace tensorflow;

REGISTER_OP("ComputeSortProj3body")
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
    .Output("threebodyonbasis: double");





class ComputeSortProj3bodyOp : public OpKernel {
 public:
  explicit ComputeSortProj3bodyOp(OpKernelConstruction* context) : OpKernel(context) {}

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



    auto input1 = input_tensor0.flat<double>();
    auto input2 = input_tensor1.flat<double>();
    auto input3 = input_tensor2.flat<int>();
    auto input4 = input_tensor3.flat<int>();
    auto input5 = input_tensor4.flat<int>();
    auto input6 = input_tensor5.flat<int>();
    auto input7 = input_tensor6.flat<int>();
    auto input8 = input_tensor7.flat<int>();
    auto input9 = input_tensor8.flat<double>();
    auto input10 = input_tensor9.flat<int>();
    //working embedding
    auto input11 = input_tensor10.flat<double>();
    auto input12 = input_tensor11.flat<int>();
    auto input13 = input_tensor12.flat<int>();

    //Copio i tensori input in nuovi array per elaborarli
    int dimbat=input7(0);
    int numdes2body=input4(0);
    int numdes3body=input3(0);
    int N=input6(0);
    int dimdes2body=dimbat*numdes2body*N;
    int dimdes3body=dimbat*numdes3body*N;
    int nsmooth_a=input10(0);

    //working type emebedding
    int nt=input12(0);
    double** type_emb3b=(double**)calloc(nt,sizeof(double*));
    for (int k=0;k<nt;k++){
        type_emb3b[k]=(double*)calloc(nt,sizeof(double));
       }
    int k=0;
    for (int i = 0; i < nt; i++) {
        for (int j = i; j < nt; j++){
            type_emb3b[i][j]=input11(k);
            type_emb3b[j][i]=input11(k);
            k=k+1;
          }
    }
    int* type_map;
    type_map=(int*)malloc(sizeof(int)*N);
    for (int i = 0; i < N; i++) {
    type_map[i]=input13(i);
    }


    //Leggo il descrittore di rete
    double* radiale;
    radiale=(double*)malloc(sizeof(double)*dimdes2body);
    for (int i = 0; i < dimdes2body; i++) {
    radiale[i]=input2(i);
    }
    //Leggo il descrittore di rete 3body
    double* angolare;
    angolare=(double*)malloc(sizeof(double)*dimdes3body);
    for (int i = 0; i < dimdes3body; i++) {
    angolare[i]=input1(i);
    }
    //Leggo il numero di vicini nel tempo
    int* numneigh3body;
    numneigh3body=(int*)malloc(sizeof(int)*dimbat*N*(numdes3body*2+1));
    for (int k=0;k<dimbat*N*(numdes3body*2+1);k++){
        numneigh3body[k]=input5(k);
    }
    int* numneigh2body;
    numneigh2body=(int*)malloc(sizeof(int)*dimbat*N*(numdes2body+1));
    for (int k=0;k<dimbat*N*(numdes2body+1);k++){
        numneigh2body[k]=input8(k);
    }

    //Alloco i beta
    double* smooth_a=(double*)malloc(nsmooth_a*3*sizeof(double));
    for (int i=0; i< nsmooth_a*3; i++){
        smooth_a[i]=input9(i);
        }

    //Alloco l'array di output
    double* smooth3body=(double*)calloc(nsmooth_a*N*dimbat,sizeof(double));


    //Calcolo della proiezione su base
    computeSmoothMax3body(radiale,angolare,numdes2body,numdes3body,
                          smooth3body,dimbat,N,numneigh3body,smooth_a,nsmooth_a,
                          type_emb3b,nt,type_map);


    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (nsmooth_a);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();

    //Copio element per elemento l'array di output nel tensore di output
    for (int i = 0; i < (dimbat*N*nsmooth_a); i++) {
    output_flat(i)=smooth3body[i];
    }
    free(smooth3body);
    free(smooth_a);
    free(radiale);
    free(angolare);
    free(numneigh3body);
    free(numneigh2body);
    free(type_map);
    for (int k=0;k<nt;k++){
         free(type_emb3b[k]);
    }
    free(type_emb3b);
  }
};
REGISTER_KERNEL_BUILDER(Name("ComputeSortProj3body").Device(DEVICE_CPU), ComputeSortProj3bodyOp);

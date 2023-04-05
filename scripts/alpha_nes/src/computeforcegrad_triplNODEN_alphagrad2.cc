#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>

//DA CONTROLLAREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE LA BARTE BUFF_A1_RAD


unsigned int OMPThreadsNum=8;
void setOMPthreads(unsigned int num){OMPThreadsNum=num;};

void computenextgrad_tripl(double* prevgrad, double* desr, double* desa,
  double* intderiv_r, double* intderiv_a, int* intmap_r, int* intmap_a, int nr,
  int na, int N, int dimbat,double* nextgrad, double* smooth_a, int nsmooth_a, double* nextgrad2,
  double* NG,double** nextgrad_emb3b,double** type_emb3b,int nt,int* type_map){

   #pragma omp parallel for num_threads(OMPThreadsNum)
   for (int b=0; b<dimbat; b++){
       for (int par=0; par<N; par++){
           int nne3bod=intmap_a[b*N*(na*2+1)+par*(na*2+1)];
           int na_real=nne3bod*(nne3bod-1)/2;
           int actual=b*N*nr+par*nr;
           int actual_ang=b*N*na+par*na;
           int actgrad=b*N*nsmooth_a+par*nsmooth_a;
           int actgrad2=b*N*nsmooth_a*3+par*nsmooth_a*3;
           int nn=0;
           for (int j=0;j<nne3bod-1;j++){
               for (int k=j+1;k<nne3bod;k++){
                    int neighj=intmap_a[b*(N*(na*2+1))+(na*2+1)*par+nn*2+1];
                    int neighk=intmap_a[b*(N*(na*2+1))+(na*2+1)*par+nn*2+2];


                    int j_type=type_map[neighj];
                    int k_type=type_map[neighk];
                    double chtjk_par=type_emb3b[j_type][k_type];

                    double angulardes=desa[actual_ang+nn];
                    double radialdes_j=desr[actual+j];
                    double radialdes_k=desr[actual+k];

                    for (int cor=0; cor<3; cor++){
                       double intder_j=intderiv_a[b*(N*na)*3*2+par*na*3*2+cor*na*2+nn*2];
                       double intder_k=intderiv_a[b*(N*na)*3*2+par*na*3*2+cor*na*2+nn*2+1];

                       double intder_r_j=intderiv_r[b*N*3*nr+nr*3*par+cor*nr+j];
                       double intder_r_k=intderiv_r[b*N*3*nr+nr*3*par+cor*nr+k];

                       double prevgrad_loc=prevgrad[b*(N*3)+3*par+cor];
                       double prevgrad_neighj=prevgrad[b*(N*3)+3*neighj+cor];
                       double prevgrad_neighk=prevgrad[b*(N*3)+3*neighk+cor];


                       for (int a1=0; a1<nsmooth_a; a1++){
                           double beta=smooth_a[a1*3+2];
                           double alpha1=smooth_a[a1*3];
                           double alpha2=smooth_a[a1*3+1];

                           double expbeta=exp(beta*angulardes);

                           double sim1=exp(alpha2*radialdes_j+alpha1*radialdes_k);
                           double sim2=exp(alpha1*radialdes_j+alpha2*radialdes_k);
                           double sum=sim1+sim2;

                           double delta=expbeta*(1.+beta*angulardes)*sum*0.5;

                           double suppj=(alpha1*sim2+alpha2*sim1)*expbeta;
                           double suppk=(alpha1*sim1+alpha2*sim2)*expbeta;
                           double Bp_j=suppj*angulardes*0.5;
                           double Bp_k=suppk*angulardes*0.5;

                           double gradxij=chtjk_par*delta*intder_j+chtjk_par*Bp_j*intder_r_j;
                           double gradxik=chtjk_par*delta*intder_k+chtjk_par*Bp_k*intder_r_k;
                           nextgrad[actgrad+a1]-=prevgrad_loc*0.5*(gradxij+gradxik);
                           nextgrad[actgrad+a1]+=prevgrad_neighj*0.5*gradxij;
                           nextgrad[actgrad+a1]+=prevgrad_neighk*0.5*gradxik;

                           double grad_emb_xij=(delta*intder_j+Bp_j*intder_r_j);
                           double grad_emb_xik=(delta*intder_k+Bp_k*intder_r_k);

                           double NGel=NG[b*N*nsmooth_a+par*nsmooth_a+a1];



                           nextgrad_emb3b[j_type][k_type]-=prevgrad_loc*0.5*NGel*(grad_emb_xij+grad_emb_xik);
                           nextgrad_emb3b[j_type][k_type]+=prevgrad_neighj*0.5*NGel*grad_emb_xij;
                           nextgrad_emb3b[j_type][k_type]+=prevgrad_neighk*0.5*NGel*grad_emb_xik;

                           nextgrad_emb3b[k_type][j_type]-=prevgrad_loc*0.5*NGel*(grad_emb_xij+grad_emb_xik);
                           nextgrad_emb3b[k_type][j_type]+=prevgrad_neighj*0.5*NGel*grad_emb_xij;
                           nextgrad_emb3b[k_type][j_type]+=prevgrad_neighk*0.5*NGel*grad_emb_xik;



                           double buff_a1_ang=expbeta*(1.+beta*angulardes)*(sim1*radialdes_k+sim2*radialdes_j)*0.5;
                           double buff_a2_ang=expbeta*(1.+beta*angulardes)*(sim1*radialdes_j+sim2*radialdes_k)*0.5;
                           double buff_beta_ang=expbeta*angulardes*(2.+beta*angulardes)*sum*0.5;

                           double buff_beta_r_j=suppj*angulardes*angulardes*0.5;
                           double buff_beta_r_k=suppk*angulardes*angulardes*0.5;

			   double buff_a1_r_j=(sim2+alpha1*sim2*radialdes_j+alpha2*sim1*radialdes_k)*expbeta*0.5*angulardes;
                           double buff_a2_r_j=(sim1+alpha2*sim1*radialdes_j+alpha1*sim2*radialdes_k)*expbeta*0.5*angulardes;

                           double buff_a1_r_k=(sim1+alpha1*sim1*radialdes_k+alpha2*sim2*radialdes_j)*expbeta*0.5*angulardes;
                           double buff_a2_r_k=(sim2+alpha2*sim2*radialdes_k+alpha1*sim1*radialdes_j)*expbeta*0.5*angulardes;

                           double grad_a1_xij=chtjk_par*buff_a1_ang*intder_j+chtjk_par*buff_a1_r_j*intder_r_j;
                           double grad_a1_xik=chtjk_par*buff_a1_ang*intder_k+chtjk_par*buff_a1_r_k*intder_r_k;

                           double grad_a2_xij=chtjk_par*buff_a2_ang*intder_j+chtjk_par*buff_a2_r_j*intder_r_j;
                           double grad_a2_xik=chtjk_par*buff_a2_ang*intder_k+chtjk_par*buff_a2_r_k*intder_r_k;

                           double grad_beta_xij=chtjk_par*buff_beta_ang*intder_j+chtjk_par*buff_beta_r_j*intder_r_j;
                           double grad_beta_xik=chtjk_par*buff_beta_ang*intder_k+chtjk_par*buff_beta_r_k*intder_r_k;




                           nextgrad2[a1*3]-=prevgrad_loc*0.5*NGel*(grad_a1_xij+grad_a1_xik);
                           nextgrad2[a1*3]+=prevgrad_neighj*0.5*NGel*grad_a1_xij;
                           nextgrad2[a1*3]+=prevgrad_neighk*0.5*NGel*grad_a1_xik;

                           nextgrad2[a1*3+1]-=prevgrad_loc*0.5*NGel*(grad_a2_xij+grad_a2_xik);
                           nextgrad2[a1*3+1]+=prevgrad_neighj*0.5*NGel*grad_a2_xij;
                           nextgrad2[a1*3+1]+=prevgrad_neighk*0.5*NGel*grad_a2_xik;

                           nextgrad2[a1*3+2]-=prevgrad_loc*0.5*NGel*(grad_beta_xij+grad_beta_xik);
                           nextgrad2[a1*3+2]+=prevgrad_neighj*0.5*NGel*grad_beta_xij;
                           nextgrad2[a1*3+2]+=prevgrad_neighk*0.5*NGel*grad_beta_xik;


                       }
                   }
                   nn=nn+1;

        }
      }
    }
  }
  for (int k=0;k<nt;k++){
      nextgrad_emb3b[k][k]*=0.5;
    }
}
















using namespace tensorflow;

REGISTER_OP("ComputeForceTriplGrad")
    .Input("prevgrad: double")
    .Input("netderiv: double")
    .Input("desr: double")
    .Input("desa: double")
    .Input("intder_r: double")
    .Input("intder_a: double")
    .Input("intmap_r: int32")
    .Input("intmap_a: int32")
    .Input("numdesr: int32")
    .Input("numdesa: int32")
    .Input("numpar: int32")
    .Input("dimbatch: int32")
    .Input("nsmooth_a: int32")
    .Input("smooth_a: double")
    .Input("type_emb3b: double")
    .Input("nt_couple: int32")
    .Input("type_map: int32")
    .Output("nextgrad_ng: double")
    .Output("nextgrad_alpha: double")
    .Output("next_emb3b_grad: double");
class ComputeForceTriplGradOp : public OpKernel {
 public:
  explicit ComputeForceTriplGradOp(OpKernelConstruction* context) : OpKernel(context) {}
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
    const Tensor& input_tensor13 = context->input(12);
    const Tensor& input_tensor14 = context->input(13);

    const Tensor& input_tensor15 = context->input(14);
    const Tensor& input_tensor16 = context->input(15);
    const Tensor& input_tensor17 = context->input(16);


    //flatting the tensor
    auto input1 = input_tensor1.flat<double>();
    auto input2 = input_tensor2.flat<double>();
    auto input3 = input_tensor3.flat<double>();
    auto input4 = input_tensor4.flat<double>();
    auto input5 = input_tensor5.flat<double>();
    auto input6 = input_tensor6.flat<double>();

    auto input7 = input_tensor7.flat<int>();
    auto input8 = input_tensor8.flat<int>();
    auto input9 = input_tensor9.flat<int>();
    auto input10 = input_tensor10.flat<int>();
    auto input11 = input_tensor11.flat<int>();
    auto input12 = input_tensor12.flat<int>();

    auto input13 = input_tensor13.flat<int>();
    auto input14 = input_tensor14.flat<double>();
    auto input15 = input_tensor15.flat<double>();
    auto input16 = input_tensor16.flat<int>();
    auto input17 = input_tensor17.flat<int>();


    //COPYING ALL TENSOR IN ARRAY TO OPERATE ON THEM

    //dimansion parameters
    int nsmooth_a=input13(0);
    int dimbat=input12(0);
    int N=input11(0);
    int na=input10(0);
    int nr=input9(0);
    int netderivdim=dimbat*N*nsmooth_a;


    ///Working on emebdding
    int nt=input16(0);
    int nt_couple=nt+nt*(nt-1)/2;


    double** type_emb3b=(double**)calloc(nt,sizeof(double*));
    for (int k=0;k<nt;k++){
        type_emb3b[k]=(double*)calloc(nt,sizeof(double));
       }
    int k=0;
    for (int i = 0; i < nt; i++) {
        for (int j = i; j < nt; j++){
            type_emb3b[i][j]=input15(k);
            type_emb3b[j][i]=input15(k);
            k=k+1;
          }
    }
    int* type_map;
    type_map=(int*)calloc(N,sizeof(int));
    for (int i = 0; i < N; i++) {
    type_map[i]=input17(i);
    }

   double* NG;
   NG=(double*)malloc(sizeof(double)*dimbat*N*nsmooth_a);
   for (int i = 0; i < dimbat*N*nsmooth_a; i++) {
    NG[i]=input2(i);
    }

    double* radiale;
    radiale=(double*)malloc(sizeof(double)*dimbat*N*nr);
    for (int i = 0; i < dimbat*N*nr; i++) {
    radiale[i]=input3(i);
    }
    //3body descriptors
    double* angolare;
    angolare=(double*)malloc(sizeof(double)*dimbat*N*na);
    for (int i = 0; i < dimbat*N*na; i++) {
    angolare[i]=input4(i);
    }
    //radial derivatives
    double* intderiv_r;
    intderiv_r=(double*)malloc(sizeof(double)*dimbat*N*nr*3);
    for (int i = 0; i < dimbat*N*nr*3; i++) {
    intderiv_r[i]=input5(i);
    }
    //3body derivatives
    double* intderiv_a;
    intderiv_a=(double*)malloc(sizeof(double)*dimbat*N*na*3*2);
    for (int i = 0; i < dimbat*N*na*3*2; i++) {
    intderiv_a[i]=input6(i);
    }
    //2body interactions
    int* intmap_r;
    intmap_r=(int*)malloc(sizeof(int)*dimbat*N*(nr+1));
    for (int i = 0; i < dimbat*N*(nr+1); i++) {
    intmap_r[i]=input7(i);
    }
    //3body interactions
    int* intmap_a;
    intmap_a=(int*)malloc(sizeof(int)*dimbat*N*(na*2+1));
    for (int i = 0; i < dimbat*N*(na*2+1); i++) {
    intmap_a[i]=input8(i);
    }
    //prevgrad
    double* prevgrad;
    prevgrad=(double*)malloc(sizeof(double)*dimbat*N*3);
    for (int i=0;i < dimbat*N*3; i++){
        prevgrad[i]=input1(i);
    }

    double* smooth_a=(double*)malloc(nsmooth_a*3*sizeof(double));
           for (int i=0; i< nsmooth_a*3; i++){
               smooth_a[i]=input14(i);
        }


    //Allocating the output vector
    double* nextgrad;
    nextgrad=(double*)malloc(sizeof(double)*netderivdim);
    for (int i=0; i<  netderivdim; i++){
        nextgrad[i]=0.;
    }
    double* nextgrad2;
    nextgrad2=(double*)malloc(sizeof(double)*nsmooth_a*3);
    for (int i=0; i<  nsmooth_a*3; i++){
        nextgrad2[i]=0.;
    }
    double** nextgrad_emb3b=(double**)calloc(nt,sizeof(double*));
    for (int k=0;k<nt;k++){
        nextgrad_emb3b[k]=(double*)calloc(nt,sizeof(double));
       }

    // Create an output tensor for gradient wrt to NG
    Tensor* nextgradw3b_1 = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (1);
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (nsmooth_a);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &nextgradw3b_1));
    auto nextgradw3bflat_1 = nextgradw3b_1->flat<double>();
    // Create an output tensor for gradient wrt to alphas
    Tensor* nextgradw3b_2 = NULL;
    TensorShape grad_net_shape2 ;
    grad_net_shape2.AddDim (nsmooth_a);
    grad_net_shape2.AddDim (3);
    OP_REQUIRES_OK(context, context->allocate_output(1, grad_net_shape2,
                                                     &nextgradw3b_2));
    auto nextgradw3bflat_2 = nextgradw3b_2->flat<double>();

    // Create an output tensor for gradient wrt to alphas
    Tensor* nextgradw3b_3 = NULL;
    TensorShape grad_net_shape3 ;
    grad_net_shape3.AddDim (nt_couple);

    OP_REQUIRES_OK(context, context->allocate_output(2, grad_net_shape3,
                                                     &nextgradw3b_3));
    auto nextgradw3bflat_3 = nextgradw3b_3->flat<double>();




    ///FINE


    //COMPUTING NEXTGRAD
   computenextgrad_tripl(prevgrad, radiale, angolare, intderiv_r, intderiv_a,
     intmap_r, intmap_a, nr, na, N, dimbat,nextgrad,smooth_a,nsmooth_a,
     nextgrad2,NG,nextgrad_emb3b,type_emb3b,nt,type_map);



    //Copying computed force in Tensorflow framework
    for (int i = 0; i < (dimbat*N*nsmooth_a); i++) {
    nextgradw3bflat_1(i)=nextgrad[i];
    }
    //Copying computed force in Tensorflow framework
    for (int i = 0; i < (nsmooth_a*3); i++) {
    nextgradw3bflat_2(i)=nextgrad2[i];
    }
    k=0;
    for (int i = 0; i < nt; i++) {
        for (int j = i; j < nt; j++){
             nextgradw3bflat_3(k)=nextgrad_emb3b[i][j];
             k=k+1;
           }
    }


    //Free memory
    free(NG);
    free(prevgrad);
    free(nextgrad);
    free(nextgrad2);
    for (int k=0;k<nt;k++){
         free(nextgrad_emb3b[k]);
    }
    free(nextgrad_emb3b);
    free(radiale);
    free(angolare);
    free(intderiv_r);
    free(intderiv_a);
    free(intmap_r);
    free(intmap_a);
    free(type_map);

    for (int k=0;k<nt;k++){
         free(type_emb3b[k]);
    }
    free(type_emb3b);
    //free alphas
    free(smooth_a);
    }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceTriplGrad").Device(DEVICE_CPU), ComputeForceTriplGradOp);

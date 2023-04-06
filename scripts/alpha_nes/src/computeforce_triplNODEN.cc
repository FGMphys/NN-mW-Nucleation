#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <math.h>







void computeforce_tripl(double* netderiv, double* desr, double* desa,
                        double* intderiv_r, double* intderiv_a, int* intmap_r,
                        int* intmap_a, int nr, int na, int N, int dimbat,
                        double* force, int nsmooth_a,double* smooth_a,
                        double** type_emb3b,int nt,int* type_map){
   double delta=0.;
   double Bp_j=0.;
   double Bp_k=0.;
   for (int b=0; b<dimbat; b++){
       for (int par=0; par<N; par++){
           int nne3bod=intmap_a[b*N*(na*2+1)+par*(na*2+1)];
           int na_real=nne3bod*(nne3bod-1)/2;
           int actual=b*N*nr+par*nr;
           int actual_ang=b*N*na+par*na;
           int actgrad=b*N*nsmooth_a+par*nsmooth_a;
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


                       for (int a1=0; a1<nsmooth_a; a1++){
                           double beta=smooth_a[a1*3+2];
                           double alpha1=smooth_a[a1*3];
                           double alpha2=smooth_a[a1*3+1];
                           double net_der=0.5*netderiv[actgrad+a1]*chtjk_par;

                           double expbeta=exp(beta*angulardes);

                           double sim1=exp(alpha2*radialdes_j+alpha1*radialdes_k);
                           double sim2=exp(alpha1*radialdes_j+alpha2*radialdes_k);
                           double sum=sim1+sim2;

                           delta=expbeta*(1.+beta*angulardes)*sum*0.5;

                           double suppj=(alpha1*sim2+alpha2*sim1)*expbeta*0.5;
                           double suppk=(alpha1*sim1+alpha2*sim2)*expbeta*0.5;
                           Bp_j=suppj*angulardes;
                           Bp_k=suppk*angulardes;

                           double fxij=net_der*(delta*intder_j+Bp_j*intder_r_j);
                           double fxik=net_der*(delta*intder_k+Bp_k*intder_r_k);

                           force[b*N*3+par*3+cor]-=(fxij+fxik);
                           force[b*N*3+neighj*3+cor]+=fxij;
                           force[b*N*3+neighk*3+cor]+=fxik;
                         }
                     }
                     nn=nn+1;
                 }
           }
        }
    }

}


















using namespace tensorflow;

REGISTER_OP("ComputeForceTripl")
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
    .Output("force: double");

class ComputeForceTriplOp : public OpKernel {
 public:
  explicit ComputeForceTriplOp(OpKernelConstruction* context) : OpKernel(context) {}
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

    //working emb3b forces
    const Tensor& input_tensor14 = context->input(13);
    const Tensor& input_tensor15 = context->input(14);
    const Tensor& input_tensor16 = context->input(15);


    //flatting the tensor
    auto input1 = input_tensor1.flat<double>();
    auto input2 = input_tensor2.flat<double>();
    auto input3 = input_tensor3.flat<double>();
    auto input4 = input_tensor4.flat<double>();
    auto input5 = input_tensor5.flat<double>();
    auto input6 = input_tensor6.flat<int>();
    auto input7 = input_tensor7.flat<int>();
    auto input8 = input_tensor8.flat<int>();
    auto input9 = input_tensor9.flat<int>();
    auto input10 = input_tensor10.flat<int>();
    auto input11 = input_tensor11.flat<int>();
    auto input12 = input_tensor12.flat<int>();
    auto input13 = input_tensor13.flat<double>();

    //working emb3b forces
    auto input14 = input_tensor14.flat<double>();
    auto input15 = input_tensor15.flat<int>();
    auto input16 = input_tensor16.flat<int>();


    //dimansion parameters
    int dimbat=input11(0);
    int N=input10(0);
    int nsmooth_a=input12(0);
    int na=input9(0);
    int nr=input8(0);
    int netderivdim=dimbat*N*nsmooth_a;

    //working type emebedding
    int nt=input15(0);
    double** type_emb3b=(double**)calloc(nt,sizeof(double*));
    for (int k=0;k<nt;k++){
        type_emb3b[k]=(double*)calloc(nt,sizeof(double));
       }
    int k=0;
    for (int i = 0; i < nt; i++) {
        for (int j = i; j < nt; j++){
            type_emb3b[i][j]=input14(k);
            type_emb3b[j][i]=input14(k);
            k=k+1;
          }
    }
    int* type_map;
    type_map=(int*)malloc(sizeof(int)*N);
    for (int i = 0; i < N; i++) {
    type_map[i]=input16(i);
    }

    double* net_deriv;
    net_deriv=(double*)malloc(sizeof(double)*netderivdim);
    for (int i = 0; i < netderivdim; i++) {
    net_deriv[i]=input1(i);
    }
    //radial descriptors
    double* radiale;
    radiale=(double*)malloc(sizeof(double)*dimbat*N*nr);
    for (int i = 0; i < dimbat*N*nr; i++) {
    radiale[i]=input2(i);
    }
    //3body descriptors
    double* angolare;
    angolare=(double*)malloc(sizeof(double)*dimbat*N*na);
    for (int i = 0; i < dimbat*N*na; i++) {
    angolare[i]=input3(i);
    }
    //radial derivatives
    double* intderiv_r;
    intderiv_r=(double*)malloc(sizeof(double)*dimbat*N*nr*3);
    for (int i = 0; i < dimbat*N*nr*3; i++) {
    intderiv_r[i]=input4(i);
    }
    //3body derivatives
    double* intderiv_a;
    intderiv_a=(double*)malloc(sizeof(double)*dimbat*N*na*3*2);
    for (int i = 0; i < dimbat*N*na*3*2; i++) {
    intderiv_a[i]=input5(i);
    }
    //2body interactions
    int* intmap_r;
    intmap_r=(int*)malloc(sizeof(int)*dimbat*N*(nr+1));
    for (int i = 0; i < dimbat*N*(nr+1); i++) {
    intmap_r[i]=input6(i);
    }
    //3body interactions
    int* intmap_a;
    intmap_a=(int*)malloc(sizeof(int)*dimbat*N*(na*2+1));
    for (int i = 0; i < dimbat*N*(na*2+1); i++) {
    intmap_a[i]=input7(i);
    }

    double* smooth_a=(double*)malloc(nsmooth_a*3*sizeof(double));
    for (int i=0; i< nsmooth_a*3; i++){
        smooth_a[i]=input13(i);
        }



    //Allocating buffers and vectors for computing hard derivatives
    //WARNING: Allocation is going to be made, but this kind of allocation is valid only
    //for the training procedure cause we know that number of neighbours never exceeds the
    // the buffer of radial descriptors. Consider other allocation for Molecular Dynamics!


    //Allocating the output vector
    double* force;
    force=(double*)malloc(sizeof(double)*dimbat*N*3);
    for (int i=0; i< dimbat*N*3; i++){
        force[i]=0.;
    }

    // Create an output tensor for forces
    Tensor* forces3b = NULL;
    TensorShape grad_net_shape ;
    grad_net_shape.AddDim (dimbat);
    grad_net_shape.AddDim (N);
    grad_net_shape.AddDim (3);
    OP_REQUIRES_OK(context, context->allocate_output(0, grad_net_shape,
                                                     &forces3b));
    auto forces3bflat = forces3b->flat<double>();


    //COMPUTING FORCES
   computeforce_tripl(net_deriv, radiale, angolare, intderiv_r, intderiv_a,
                      intmap_r, intmap_a, nr, na, N, dimbat, force,
                      nsmooth_a,smooth_a,type_emb3b,nt,type_map);



    //Copying computed force in Tensorflow framework
    for (int i = 0; i < (dimbat*N*3); i++) {
    forces3bflat(i)=force[i];
    }



    //Free memory
    free(force);
    free(radiale);
    free(angolare);
    free(net_deriv);
    free(intderiv_r);
    free(intderiv_a);
    free(intmap_r);
    free(intmap_a);
    //free alphas
    free(smooth_a);
    //free emb3b
    free(type_map);
    for (int k=0;k<nt;k++){
         free(type_emb3b[k]);
    }
    free(type_emb3b);
    }
};
REGISTER_KERNEL_BUILDER(Name("ComputeForceTripl").Device(DEVICE_CPU), ComputeForceTriplOp);

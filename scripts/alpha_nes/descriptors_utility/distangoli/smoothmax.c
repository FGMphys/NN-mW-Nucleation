#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <ctype.h>


#include "vector.h"
#include "global_definitions.h"
#include "interaction_map.h"
#include "utilities.h"
#include "cell_list.h"
#include "smart_allocator.h"

#define SQR(x) ((x)*(x))
#define Sqrt(x) (sqrt(x))
#define Power(x,n) (pow(x,n))

#define MAX_LINE_LENGTH 2000
#define MAX_NNEIGHBOURS 300

typedef struct _distsymm {
  int index;
  double dist;
  double dx;
  double dy;
  double dz;
} distsymm;

typedef struct _distangle {
  int indexj;
  int indexk;
  double distj;
  double distk;
  double angle;
  double dxij;
  double dyij;
  double dzij;
  double dxik;
  double dyik;
  double dzik;
} distangle;

int getLine(char *line,FILE *pfile)
{
	if (fgets(line,MAX_LINE_LENGTH,pfile) == NULL)
		return 0;
	else
		return strlen(line);
}

int readNumSpheres(char *nomefile)
{
  char line[MAX_LINE_LENGTH]="";

  FILE *pfile=fopen(nomefile,"r");

  int numlines=-1;

  while (getLine(line,pfile)>0)
  {
    numlines++;
  }

  fclose(pfile);

  return numlines;

}

void readPosSpheres(char *nomefile,int num,vector *pos,double *box,double *inobox,steps *time)
{
  char line[MAX_LINE_LENGTH]="";

  FILE *pfile=fopen(nomefile,"r");

  // PRIMA RIGA
  int n;
  getLine(line,pfile);

  char stringa[500];
  strcpy(stringa,line);
  char *pch;
  int tokens=0;
  pch = strtok (stringa," ");
  while (pch != NULL)
  {
    tokens++;
    pch = strtok (NULL, " ");
  }

  int state;
  if (tokens==8)
    state=sscanf(line,"%lld %d %lf %lf %lf %lf %lf %lf\n",time,&n,box+0,box+1,box+2,box+3,box+4,box+5);
  else if (tokens==5)
  {
    state=sscanf(line,"%lld %d %lf %lf %lf\n",time,&n,box+0,box+3,box+5);
    box[1]=0.;
    box[2]=0.;
    box[4]=0.;
  }
  else
  {
    printf("Error while reading header of file %s\n",nomefile);
    exit(1);
  }



  assert(n==num);


  inobox[0]=1./box[0];
	inobox[1]=-box[1]/(box[0]*box[3]);
	inobox[2]=(box[1]*box[4])/(box[0]*box[3]*box[5])-box[2]/(box[0]*box[5]);
	inobox[3]=1./box[3];
	inobox[4]=-box[4]/(box[3]*box[5]);
	inobox[5]=1./box[5];



  int i=0;
  while (getLine(line,pfile)>0)
  {
    vector p;
    sscanf(line,"%lf %lf %lf",&p.x,&p.y,&p.z);


    pos[i].x=(inobox[0]*p.x+inobox[1]*p.y+inobox[2]*p.z);
		pos[i].y=(inobox[3]*p.y+inobox[4]*p.z);
		pos[i].z=(inobox[5]*p.z);

    i++;
  }

  assert(i==num);

  fclose(pfile);

}

void readCodeFile(char *nomefile,int *code,int num)
{
  FILE *pfile=fopen(nomefile,"r");
  if (pfile==NULL)
  {
    printf("Error: cannot open code file %s",nomefile);
    exit(1);
  }
  else
  {
    int i=0;
    while (fscanf(pfile,"%d\n",code+i)!=EOF)
      i++;
    assert(i==num);
  }

}


int compareDouble(const void * a, const void * b)
{
  if ( *(double*)a <  *(double*)b ) return -1;
  else if ( *(double*)a == *(double*)b ) return 0;
  else return 1;
}


int compare_distsymm(const void * a, const void * b)
{

  if ( ((distsymm*)a)->dist <  ((distsymm*)b)->dist )
    return -1;
  else
    return 1;

}

// in realtà sono coseni
int compare_distangle(const void * a, const void * b)
{

  if ( ((distangle*)a)->angle >  ((distangle*)b)->angle )
    return -1;
  else
    return 1;

}


int insertionSortGeneric(void *array,void *element,void *buffer,int *length,size_t size,int (*compar)(const void*,const void*))
{
	int l=*length;

  memcpy(array+l*size,element,size);

	while ((l>0) && (compar(array+(l-1)*size,array+l*size)==1))
	{
    memcpy(buffer,array+l*size,size);
    memcpy(array+l*size,array+(l-1)*size,size);
    memcpy(array+(l-1)*size,buffer,size);

		l--;
	}
	(*length)++;

  return l;
}

void buildUniformAlpha(double *alpha,int num_alpha,double alpha_mod_max)
{
  int i;
  for (i=0;i<num_alpha;i++)
  {
    alpha[i]=alpha_mod_max*(2.*i/(double)num_alpha-1.);
  }
}

void computeSmoothMax(double *ds,int num_ds,double *alpha,int num_alpha,double *buffer1,double *buffer2,double *sds,double **sds_deriv)
{
  int i;

  // costruiamo i descrittori

  for (i=0;i<num_alpha;i++)
  {
    buffer1[i]=0.;
    buffer2[i]=0.;

    int j;
    for (j=0;j<num_ds;j++)
    {
      buffer1[i]+=ds[j]*exp(alpha[i]*ds[j]);
      buffer2[i]+=exp(alpha[i]*ds[j]);
    }

    sds[i]=buffer1[i]/buffer2[i];

    for (j=0;j<num_ds;j++)
    {
      sds_deriv[i][j]=((exp(alpha[i]*ds[j])*(1.+alpha[i]*ds[j]))/buffer2[i])-alpha[i]*exp(alpha[i]*ds[j])*buffer1[i]/(buffer2[i]*buffer2[i]);
    }
  }
}


int main(int argc,char *argv[])
{

	if (argc!=7)
	{
		printf("%s [input file] [range] [nn input size] [range angolare] [nn angular size] [output suffix]\n",argv[0]);
		exit(1);
	}

  char input[100];
	strcpy(input,argv[1]);
  double range=atof(argv[2]);
  int n_input=atoi(argv[3]);
  double range_angolare=atof(argv[4]);
  int n_input_angolare=atoi(argv[5]);
  char suffix[100];
  strcpy(suffix,argv[6]);


  // determiniamo se code e' un numero o un file di numeri
  int i;

  // ALLOCATIONS
  int num=readNumSpheres(input);
  vector *pos=calloc(num,sizeof(vector));


  // READ POSITIONS
  steps time;
  double box[6];
  double inobox[6];
  readPosSpheres(input,num,pos,box,inobox,&time);

  // GENERATE CELLS
  listcell *cells=getList(box,range,num);
	fullUpdateList(cells,pos,num,box,range);

  // INTERACTION MAPS
  interactionmap *ime=createInteractionMap(num,MAX_NNEIGHBOURS);
  interactionmap *im=createInteractionMap(num,MAX_NNEIGHBOURS);

  calculateInteractionMapWithCutoffDistanceOrdered(cells,ime,pos,box,range);
  buildImFromIme(ime,im);

  // struttura per il parametro d'ordine
  distsymm **ds=calloc(num,sizeof(distsymm*));
  distangle **da=calloc(num,sizeof(distangle*));
  int *ds_num=calloc(num,sizeof(int));
  int *da_num=calloc(num,sizeof(int));
  int *ds_num_angular=calloc(num,sizeof(int));

  double *radiale=calloc(n_input,sizeof(double));
  double *angolare=calloc(n_input_angolare,sizeof(double));
  int num_alpha_radiale=n_input;
  int num_alpha_angolare=n_input_angolare;
  double alpha_radiale_modmax=100.;
  double alpha_angolare_modmax=1000000.;
  int max_input=(num_alpha_radiale>num_alpha_angolare?num_alpha_radiale:num_alpha_angolare);
  double *buffer1=calloc(max_input,sizeof(double));
  double *buffer2=calloc(max_input,sizeof(double));
  double *alpha_radiale=calloc(num_alpha_radiale,sizeof(double));
  double *alpha_angolare=calloc(num_alpha_angolare,sizeof(double));
  double *smoothradiale=calloc(num_alpha_radiale,sizeof(double));
  double *smoothangolare=calloc(num_alpha_angolare,sizeof(double));
  double **smoothradialederiv=calloc(num_alpha_radiale,sizeof(double*));
  double **smoothangolarederiv=calloc(num_alpha_angolare,sizeof(double*));


  for (i=0;i<num;i++)
  {
    ds[i]=calloc(1+n_input,sizeof(distsymm));
    da[i]=calloc(1+n_input_angolare,sizeof(distangle));

    smoothradialederiv[i]=calloc(n_input,sizeof(double));
    smoothangolarederiv[i]=calloc(n_input_angolare,sizeof(double));


    ds_num[i]=0;
    ds_num_angular[i]=0;
    da_num[i]=0;
  }

  // I - PRIMI VICINI //////////////////////////////////////////////////////////

  for (i=0;i<num;i++)
  {
    distsymm dij,buffer;
    int im_index=0;
    vector olddist,dist;
    int pos_index;

    while (im_index<im->howmany[i])
    {

      int j=im->with[i][im_index];

      dij.index=j;
      //dij.dist2=im->rij2[i][im_index];

      // calcolo per le derivate
      olddist.x=pos[i].x-pos[j].x;
      olddist.y=pos[i].y-pos[j].y;
      olddist.z=pos[i].z-pos[j].z;

      olddist.x-=rint(olddist.x);
      olddist.y-=rint(olddist.y);
      olddist.z-=rint(olddist.z);

      dist.x=box[0]*olddist.x+box[1]*olddist.y+box[2]*olddist.z;
      dist.y=box[3]*olddist.y+box[4]*olddist.z;
      dist.z=box[5]*olddist.z;

      dij.dist=sqrt(SQR(dist.x)+SQR(dist.y)+SQR(dist.z));

      dij.dx=dist.x;
      dij.dy=dist.y;
      dij.dz=dist.z;


      pos_index=insertionSortGeneric(ds[i],&dij,&buffer,ds_num+i,sizeof(distsymm),compare_distsymm);

      if (ds_num[i]>n_input)
        ds_num[i]=n_input;
      else
      {
        if (dij.dist<range_angolare)
        {
          ds_num_angular[i]++;
        }
      }

      dij.index=i;

      dij.dx*=-1;
      dij.dy*=-1;
      dij.dz*=-1;

      pos_index=insertionSortGeneric(ds[j],&dij,&buffer,ds_num+j,sizeof(distsymm),compare_distsymm);

      if (ds_num[j]>n_input)
        ds_num[j]=n_input;
      else
      {
        if (dij.dist<range_angolare)
        {
          ds_num_angular[j]++;
        }
      }

      im_index++;

    }

  }

  // calcolo della parte angolare dei descrittori
  for (i=0;i<num;i++)
  {
    int j,k;

    distangle da_ijk,buffer;
    vector olddist,dist;

    for (j=0;j<ds_num_angular[i]-1;j++)
    {
      for (k=j+1;k<ds_num_angular[i];k++)
      {

        da_ijk.indexj=ds[i][j].index;
        da_ijk.indexk=ds[i][k].index;
        da_ijk.distj=ds[i][j].dist;
        da_ijk.distk=ds[i][k].dist;

        double angle=(ds[i][j].dx*ds[i][k].dx+ds[i][j].dy*ds[i][k].dy+ds[i][j].dz*ds[i][k].dz)/(da_ijk.distj*da_ijk.distk);

        double cutoffj=SQR(1.-(da_ijk.distj/range_angolare));
        double cutoffk=SQR(1.-(da_ijk.distk/range_angolare));

        da_ijk.angle=0.5*(angle+1)*cutoffj*cutoffk;

        double xij=ds[i][j].dx;
        double yij=ds[i][j].dy;
        double zij=ds[i][j].dz;
        double dij2=SQR(ds[i][j].dist);
        double xik=ds[i][k].dx;
        double yik=ds[i][k].dy;
        double zik=ds[i][k].dz;
        double dik2=SQR(ds[i][k].dist);

        double dcij=2*(da_ijk.distj-range_angolare)/(range_angolare*range_angolare*da_ijk.distj);
        double dcxij=dcij*ds[i][j].dx;
        double dcyij=dcij*ds[i][j].dy;
        double dczij=dcij*ds[i][j].dz;

        double dcik=2*(da_ijk.distk-range_angolare)/(range_angolare*range_angolare*da_ijk.distk);
        double dcxik=dcik*ds[i][k].dx;
        double dcyik=dcik*ds[i][k].dy;
        double dczik=dcik*ds[i][k].dz;

        double danglexij=0.5*(dij2*xik - xij*(xij*xik + yij*yik + zij*zik))/(Power(dij2,1.5)*Sqrt(dik2));
        double dangleyij=0.5*(dij2*yik - yij*(xij*xik + yij*yik + zij*zik))/(Power(dij2,1.5)*Sqrt(dik2));
        double danglezij=0.5*(dij2*zik - zij*(xij*xik + yij*yik + zij*zik))/(Power(dij2,1.5)*Sqrt(dik2));

        double danglexik=0.5*(dik2*xij - xik*(xij*xik + yij*yik + zij*zik))/(Sqrt(dij2)*Power(dik2,1.5));
        double dangleyik=0.5*(dik2*yij - yik*(xij*xik + yij*yik + zij*zik))/(Sqrt(dij2)*Power(dik2,1.5));
        double danglezik=0.5*(dik2*zij - zik*(xij*xik + yij*yik + zij*zik))/(Sqrt(dij2)*Power(dik2,1.5));

        da_ijk.dxij=danglexij*cutoffj*cutoffk+0.5*(angle+1)*dcxij*cutoffk;
        da_ijk.dyij=dangleyij*cutoffj*cutoffk+0.5*(angle+1)*dcyij*cutoffk;
        da_ijk.dzij=danglezij*cutoffj*cutoffk+0.5*(angle+1)*dczij*cutoffk;

        da_ijk.dxik=danglexik*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dcxik;
        da_ijk.dyik=dangleyik*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dcyik;
        da_ijk.dzik=danglezik*cutoffj*cutoffk+0.5*(angle+1)*cutoffj*dczik;


        int ins=insertionSortGeneric(da[i],&da_ijk,&buffer,da_num+i,sizeof(distangle),compare_distangle);

        if (da_num[i]>n_input_angolare)
        {
          da_num[i]=n_input_angolare;
        }

      }
    }

  }


  // smoothmax

  buildUniformAlpha(alpha_radiale,num_alpha_radiale,alpha_radiale_modmax);
  buildUniformAlpha(alpha_angolare,num_alpha_angolare,alpha_angolare_modmax);

  for (i=0;i<num;i++)
  {
    int j;
    for (j=0;j<ds_num[i];j++)
    {
      radiale[j]=SQR(1.-(ds[i][j].dist/range));
    }
    for (j=ds_num[i];j<n_input;j++)
    {
      radiale[j]=0.;
    }

    for (j=0;j<da_num[i];j++)
    {
      angolare[j]=da[i][j].angle;
    }
    for (j=da_num[i];j<n_input_angolare;j++)
    {
      angolare[j]=0.;
    }

    computeSmoothMax(radiale,n_input,alpha_radiale,num_alpha_radiale,buffer1,buffer2,smoothradiale,smoothradialederiv);
    computeSmoothMax(angolare,n_input_angolare,alpha_angolare,num_alpha_angolare,buffer1,buffer2,smoothangolare,smoothangolarederiv);

    /*
    int k;
    for (k=0;k<n_input;k++)
      fprintf(stderr,"%g\n",radiale[k]);
    for (k=0;k<num_alpha_radiale;k++)
      fprintf(stdout,"%g\n",smoothradiale[k]);


    for (k=0;k<n_input_angolare;k++)
      fprintf(stderr,"%g\n",angolare[k]);
    for (k=0;k<num_alpha_angolare;k++)
      fprintf(stdout,"%g\n",smoothangolare[k]);

    exit(1);
    */
  }


  // OUTPUT

  char name_descrittori[100];
  char name_derivate[100];
  char name_interazioni[100];

  char name_descrittori_angolari[100];
  char name_derivate_angolari[100];
  char name_interazioni_angolari[100];
  char name_alldes[100];

  ////////////////////////////////////////////////
  ////    INTERAZIONI                 ////////////
  ////////////////////////////////////////////////


  sprintf(name_interazioni,"interazioni%s",suffix);
  FILE *file_interazioni=fopen(name_interazioni,"w");
  fprintf(file_interazioni,"\"num_R\"");
 for (int g=0; g<n_input;g++){
      fprintf(file_interazioni,",\"IntR%d\"",g);
   }
     fprintf(file_interazioni,"\n");

  for (i=0;i<num;i++)
  {
    int j;
    int expected=n_input;
   fprintf(file_interazioni,"%d",ds_num[i]);
    for (j=0;j<ds_num[i];j++)
    {
      int neighbour=ds[i][j].index;

      fprintf(file_interazioni,",%d",neighbour);
    }
    while (j++<expected)
      fprintf(file_interazioni,",0");
    fprintf(file_interazioni,"\n");
   }

    fclose(file_interazioni);


  ////////////////////////////////////////////////
  ////    INTERAZIONI ANGOLARI        ////////////
  ////////////////////////////////////////////////


  int **angint_central,**angint_selfindex,**angint_jk;
  int *num_angint;
  Matrix2D(angint_central,num,n_input_angolare*100,int);
  Matrix2D(angint_selfindex,num,n_input_angolare*100,int);
  Matrix2D(angint_jk,num,n_input_angolare*100,int);
  num_angint=calloc(num,sizeof(int));



  for (i=0;i<num;i++)
  {
    int j;
    for (j=0;j<da_num[i];j++)
    {

      int neighbour_j=da[i][j].indexj;
      int neighbour_k=da[i][j].indexk;

      angint_central[neighbour_j][num_angint[neighbour_j]]=i;
      angint_central[neighbour_k][num_angint[neighbour_k]]=i;

      angint_selfindex[neighbour_j][num_angint[neighbour_j]]=j;
      angint_selfindex[neighbour_k][num_angint[neighbour_k]]=j;

      angint_jk[neighbour_j][num_angint[neighbour_j]]=0;
      angint_jk[neighbour_k][num_angint[neighbour_k]]=1;

      num_angint[neighbour_j]++;
      num_angint[neighbour_k]++;


    }

  }



  sprintf(name_interazioni_angolari,"interazioniangolari%s",suffix);
  FILE *file_interazioni_angolari=fopen(name_interazioni_angolari,"w");

 //Intestazione//
 fprintf(file_interazioni_angolari,"\"num_A\"");
  for (int g=0; g<n_input_angolare;g++){
      fprintf(file_interazioni_angolari,",\"Int_Aj%d\"",g);
      fprintf(file_interazioni_angolari,",\"Int_Ak%d\"",g);
   }
     fprintf(file_interazioni_angolari,"\n");


  for (i=0;i<num;i++)
  {
    int j;
    fprintf(file_interazioni_angolari,"%d",da_num[i]);
    for (j=0;j<da_num[i];j++)
    {
      fprintf(file_interazioni_angolari,",%d,%d",da[i][j].indexj,da[i][j].indexk);
    }
    while (j++<n_input_angolare)
      fprintf(file_interazioni_angolari,",0,0");
    fprintf(file_interazioni_angolari,"\n");

  }

  fclose(file_interazioni_angolari);





 //////////////////////////////
 ////TUTTI I DESCRITTORI///////
 //////////////////////////////
  sprintf(name_alldes,"alldes%s",suffix);
  FILE *file_alldes=fopen(name_alldes,"w");
  ///Intestazione
  fprintf(file_alldes,"\"dummy\"");
  for (int g=0; g<n_input;g++){
      fprintf(file_alldes,",\"Des_R%d\"",g);
   }
  for (int g=0; g<n_input_angolare;g++){
      fprintf(file_alldes,",\"Des_A%d\"",g);
   }
  fprintf(file_alldes,"\n");
  //scrittura descrittori
  for (i=0;i<num;i++)
  {
    //Descrittori radiali
    int expected=n_input;
    int l=ds_num[i];
    int li;

    // stampa descrittori



    for (li=0;li<l;li++)
    {
      double v=SQR(1.-(ds[i][li].dist/range));
      int index=ds[i][li].index;
      fprintf(file_alldes,",%14.11g",v);
    }
    while (li++<expected)
      fprintf(file_alldes,",0.");

    expected=n_input_angolare;
    l=da_num[i];
    li=0;


    for (li=0;li<l-1;li++)
    {
      fprintf(file_alldes,",%14.11g",da[i][li].angle);
    }
    fprintf(file_alldes,",%14.11g",da[i][l-1].angle);
    li++;
    while (li++<expected)
      fprintf(file_alldes,",0.");

    fprintf(file_alldes,"\n");
  }
fclose(file_alldes);


  ////////////////////////////////////////////////
  ////    DERIVATE  ANGOLARI          ////////////
  ////////////////////////////////////////////////

  sprintf(name_derivate_angolari,"derivateangolari%s",suffix);
  FILE *file_derivate_angolari=fopen(name_derivate_angolari,"w");
  ///Intestazione
  fprintf(file_derivate_angolari,"\"dummy\"");
  for (int g=0; g<n_input_angolare*2;g++){
      fprintf(file_derivate_angolari,",\"Der_A%d\"",g);
   }
  fprintf(file_derivate_angolari,"\n");
  // derivate descrittori angolari
  for (i=0;i<num;i++)
  {

    int expected=n_input_angolare;
    int l=da_num[i];
    int li;

    // stampa derivate x


    for (li=0;li<l;li++)
    {
      fprintf(file_derivate_angolari,",%14.11g,%14.11g",da[i][li].dxij,da[i][li].dxik);
    }
    while (li++<expected)
      fprintf(file_derivate_angolari,",0.,0.");

    fprintf(file_derivate_angolari,"\n");

    // stampa derivate y


    for (li=0;li<l;li++)
    {
      fprintf(file_derivate_angolari,",%14.11g,%14.11g",da[i][li].dyij,da[i][li].dyik);
    }
    while (li++<expected)
      fprintf(file_derivate_angolari,",0.,0.");

    fprintf(file_derivate_angolari,"\n");

    // stampa derivate z


    for (li=0;li<l;li++)
    {
      fprintf(file_derivate_angolari,",%14.11g,%14.11g",da[i][li].dzij,da[i][li].dzik);
    }
    while (li++<expected)
      fprintf(file_derivate_angolari,",0.,0.");

    fprintf(file_derivate_angolari,"\n");


  }


  fclose(file_derivate_angolari);


  ////////////////////////////////////////////////
  ////    DERIVATE                    ////////////
  ////////////////////////////////////////////////



  sprintf(name_derivate,"derivate%s",suffix);
  FILE *file_derivate=fopen(name_derivate,"w");
  fprintf(file_derivate,"\"dummy\"");
  for (int g=0; g<n_input;g++){
      fprintf(file_derivate,",\"Der_R%d\"",g);
   }
  fprintf(file_derivate,"\n");
  for (i=0;i<num;i++)
  {

    int expected=n_input;
    int l=ds_num[i];
    int li;

    // stampa derivate x

    for (li=0;li<l;li++)
    {
      double d=(2*(ds[i][li].dist-range)*ds[i][li].dx)/(range*range*ds[i][li].dist);
      fprintf(file_derivate,",%14.11g",d);
    }
    while (li++<expected)
      fprintf(file_derivate,",0.");

    fprintf(file_derivate,"\n");

    // stampa derivate y


    for (li=0;li<l;li++)
    {
      double d=(2*(ds[i][li].dist-range)*ds[i][li].dy)/(range*range*ds[i][li].dist);
      fprintf(file_derivate,",%14.11g",d);
    }
    while (li++<expected)
      fprintf(file_derivate,",0.");

    fprintf(file_derivate,"\n");

    // stampa derivate y


    for (li=0;li<l;li++)
    {
      double d=(2*(ds[i][li].dist-range)*ds[i][li].dz)/(range*range*ds[i][li].dist);
      fprintf(file_derivate,",%14.11g",d);
    }
    while (li++<expected)
      fprintf(file_derivate,",0.");

    fprintf(file_derivate,"\n");



  }


  fclose(file_derivate);


  return 0;

}

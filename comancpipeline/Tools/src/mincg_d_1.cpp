#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "interpolation.h"

using namespace alglib;
void function_gaussian(const real_1d_array &c, const real_1d_array &x, double &func, void *ptr) 
{
  // this callback calculates f(c,x)=exp(-c0*sqr(x0))
  // where x is a position on X-axis and c is adjustable parameter
  double K = pow(x[0]-c[1],2) + pow(x[1]-c[3],2);
  double z = exp(-0.5 * K/pow(c[2],2));
  func = c[0]*z + c[4];
}
void function_gaussian_grad(const real_1d_array &c, const real_1d_array &x, double &func, real_1d_array &grad, void *ptr) 
{
  // this callback calculates f(c,x)=exp(-c0*sqr(x0))
  // where x is a position on X-axis and c is adjustable parameter
  double K = pow(x[0]-c[1],2) + pow(x[1]-c[3],2);
  double z = exp(-0.5 * K/pow(c[2],2));
  func = c[0]*z + c[4];
  grad[0] = z;
  grad[1] =  c[0]/pow(c[2],2) * z * (x[0] - c[1]);
  grad[2] =  c[0]/pow(c[2],3) * z * K;
  grad[3] =  c[0]/pow(c[2],2) * z * (x[1] - c[3]);
  grad[4] = 1.0;
}

std::vector<double> run_fit(std::vector<double> &ix,
			    std::vector<double> &iy,
			    std::vector<double> &iz,
			    std::vector<double> &iw,
			    std::vector<double> &ic,
			    std::vector<double> &ie,
			    int &maxits,
			    double &epsx,
			    double &diffstep)
{
  ae_int_t info;
  lsfitstate state;
  lsfitreport rep;
  //int maxits=1000;
  //double epsx     = 0.000001;
  //double diffstep = 0.000001;

  real_2d_array x;
  x.setlength(ix.size(),2); 
  for (size_t i=0; i<ix.size(); i++){
    x(i,0) = ix[i];
    x(i,1) = iy[i];
  }


  real_1d_array y;
  y.setcontent(iz.size(), &iz.front());
  real_1d_array w;
  w.setcontent(iw.size(), &iw.front());
  real_1d_array c;
  c.setcontent(ic.size(), &ic.front());

  real_1d_array s = "[1,0.02,0.02,0.02,1]";
  real_1d_array steps = "[+inf,1,1,1,+inf]";

  lsfitcreatewfg(x, y, w, c, true, state);
  lsfitsetcond(state, epsx, maxits);
  //lsfitsetscale(state,s);
  //lsfitsetstpmax(state,1.0);
  alglib::lsfitfit(state, function_gaussian, function_gaussian_grad);
  lsfitresults(state, info, c, rep);

  // if (info != 0){
  //   printf("%d %f %f \n", info, iz[0], y[0]);
  //   printf("parameter fits %s \n", c.tostring(3).c_str());
  //   printf("parameter errs %s \n", rep.errpar.tostring(3).c_str());
  //   printf("Average rel error %f \n", rep.avgrelerror);
  //   printf("task condition %f \n", rep.taskrcond);
  //   printf("rms error %f \n", rep.rmserror);

  //   printf("Is grad good? %d \n", rep.varidx);
  //   printf("Iteration count? %d \n", rep.iterationscount);
  //   printf("Final error? %f \n", rep.avgerror);
  // }

  for (size_t i=0; i< ic.size(); i++){
    ic[i] = c[i];
    ie[i] = rep.errpar[i];
  }
  return ic;
}

int main(int argc, char **argv)
{
    //
    // In this example we demonstrate exponential fitting
    // by f(x) = exp(-c*x^2)
    // using function value only.
    //
    // Gradient is estimated using combination of numerical differences
    // and secant updates. diffstep variable stores differentiation step 
    // (we have to tell algorithm what step to use).
    //

  double d1[] = {-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0};
  std::vector<double> x(d1, d1 + sizeof(d1)/sizeof(double));

  double d2[] = {0.223130, 0.382893, 0.582748, 0.786628, 0.941765, 1.000000, 0.941765, 0.786628, 0.582748, 0.382893, 0.223130};
  std::vector<double> y(d2, d2 + sizeof(d2)/sizeof(double));

  double d3[] = {1,1,1,1,1,1,1,1,1,1,1};
  std::vector<double> w(d3, d3 + sizeof(d3)/sizeof(double));

  double d4[] = {0.3};
  std::vector<double> c(d4, d4 + sizeof(d4)/sizeof(double));
  ///run_fit(x,y,y,w,c,c);

  std::cout << c[0] << std::endl;
}

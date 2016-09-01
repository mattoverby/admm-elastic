// Copyright (c) 2016 University of Minnesota
// 
// lbfgssolver Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
//    conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
//    of conditions and the following disclaimer in the documentation and/or other materials
//    provided with the distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF MINNESOTA, DULUTH OR CONTRIBUTORS BE 
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
// IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Author: Ioannis Karamouzas
// Contact: ioannis@cs.umn.edu

#include <iostream>
#include <Eigen/LU>
#include "isolver.h"
#include "../linesearch/morethuente.h"

#ifndef MYLBFGSSOLVER_H_
#define MYLBFGSSOLVER_H_

namespace cppoptlib {

	/**
	* @brief  LBFGS implementation based on Nocedal & Wright Numerical Optimization book (Section 7.2)
	* @tparam T scalar type
	* @tparam P problem type
	* @tparam Ord order of solver
	*/

template<typename T>
class lbfgssolver : public ISolver<T, 1> {
public:
void minimize(Problem<T> &objFunc, Vector<T> & x0) {
	using namespace Eigen;

	int _m = std::min( int(this->settings_.maxIter), 10 );
	int _noVars = x0.size();
	double _eps_g = this->settings_.gradTol;
	double _eps_x = 1e-8;

	MatrixXd s = MatrixXd::Zero(_noVars, _m);
	MatrixXd y = MatrixXd::Zero(_noVars, _m);

	Vector<double> alpha = Vector<double>::Zero(_m);
	Vector<double> rho = Vector<double>::Zero(_m);
	Vector<double> grad(_noVars), q(_noVars), grad_old(_noVars), x_old(_noVars);

//	double f = objFunc.value(x0);
	objFunc.gradient(x0, grad);
	double gamma_k = this->settings_.init_hess;
	double gradNorm = 0;
	double alpha_init = std::min(1.0, 1.0 / grad.lpNorm<Eigen::Infinity>());
	int globIter = 0;
	int maxiter = this->settings_.maxIter;
	double new_hess_guess = 1.0; // only changed if we converged to a solution

	for (int k = 0; k < maxiter; k++)
	{
		x_old = x0;
		grad_old = grad;
		q = grad;
		globIter++;
	
		//L - BFGS first - loop recursion		
		int iter = std::min(_m, k);
		for (int i = iter - 1; i >= 0; --i)
		{
			rho(i) = 1.0 / ((s.col(i)).dot(y.col(i)));
			alpha(i) = rho(i)*(s.col(i)).dot(q);
			q = q - alpha(i)*y.col(i);
		}

		//L - BFGS second - loop recursion			
		q = gamma_k*q;
		for (int i = 0; i < iter; ++i)
		{
			double beta = rho(i)*q.dot(y.col(i));
			q = q + (alpha(i) - beta)*s.col(i);
		}

		// is there a descent
		double dir = q.dot(grad);
		if (dir < 1e-4){
			q = grad;
			maxiter -= k;
			k = 0;
			alpha_init = std::min(1.0, 1.0 / grad.lpNorm<Eigen::Infinity>() );
		}

		const double rate = MoreThuente<T, decltype(objFunc), 1>::linesearch(x0, -q,  objFunc, alpha_init) ;
//		const double rate = linesearch(objFunc, x0, -q, f, grad, 1.0);
		x0 = x0 - rate * q;
		if ((x_old - x0).squaredNorm() < _eps_x){
//			std::cout << "x diff norm: " << (x_old - x0).squaredNorm() << std::endl;
			break;
		} // usually this is a problem so exit

//		f = objFunc.value(x0);
		objFunc.gradient(x0, grad);
		
		gradNorm = grad.lpNorm<Eigen::Infinity>();
		if (gradNorm < _eps_g){
			// Only change hessian guess if we break out the loop via convergence.
			new_hess_guess = gamma_k;
			break;
		}

		Vector<double> s_temp = x0 - x_old;
		Vector<double> y_temp = grad - grad_old;

		// update the history
		if (k < _m)
		{
			s.col(k) = s_temp;
			y.col(k) = y_temp;
		}
		else
		{
			s.leftCols(_m - 1) = s.rightCols(_m - 1).eval();
			s.rightCols(1) = s_temp;
			y.leftCols(_m - 1) = y.rightCols(_m - 1).eval();
			y.rightCols(1) = y_temp;
		}
		

		gamma_k = s_temp.dot(y_temp) / y_temp.dot(y_temp);
		alpha_init = 1.0;
		
	}

	this->n_iters = globIter;
	this->settings_.init_hess = new_hess_guess;

} // end minimize

};

}
/* namespace cppoptlib */

#endif /* MYLBFGSSOLVER_H_ */

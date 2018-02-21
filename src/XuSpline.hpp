// Copyright (c) 2017, University of Minnesota
// 
// ADMM-Elastic Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
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

#ifndef ADMM_XUSPLINE_H
#define ADMM_XUSPLINE_H 1
#include <cmath>

namespace admm {

//
// 	Nonlinear Material Design Using Principal Stretches (2015)
//	Hongyi Xu, Funshing Sin, Yufeng Zhu, Jernej Barbic
//
namespace xu {

	// The spline class can be overloaded for custom splines. Otherwise,
	// some common ones (StVK, NeoHookean, Co-Rotated linear) are provided.
	class Spline {
	public:
		virtual double f(double x) const = 0;
		virtual double g(double x) const = 0;
		virtual double h(double x) const = 0;
		virtual double df(double x) const = 0;
		virtual double dg(double x) const = 0;
		virtual double dh(double x) const = 0;

		// Eq. 16: compression term that helps with stability
		static double compress_term(double kappa, double x){ return (kappa/12.0) * std::pow( (1.0-x)/6.0, 3.0 ); }
		static double d_compress_term(double kappa, double x){ return (-kappa/24.0)*( std::pow( (1.0-x)/(6.0), 2.0 ) ); }
	};

	class NeoHookean : public Spline {
	public:
		NeoHookean( double mu_, double lambda_, double kappa_ ) :
			mu(mu_), lambda(lambda_), kappa(kappa_) {}
		const double mu, lambda, kappa;
		double f(double x) const { return 0.5*mu*(x*x-1.0); }
		double g(double x) const { (void)(x); return 0.0; }
		double h(double x) const {
			double logx = std::log(x);
			return -mu*logx + 0.5*lambda*logx*logx + compress_term(kappa,x);
		}
		double df(double x) const { return mu*x; }
		double dg(double x) const { (void)(x); return 0.0; }
		double dh(double x) const { return -mu/x + lambda*std::log(x)/x + d_compress_term(kappa,x); }
	};

	class StVK : public Spline {
	public:
		StVK( double mu_, double lambda_, double kappa_ ) :
			mu(mu_), lambda(lambda_), kappa(kappa_) {}
		const double mu, lambda, kappa;
		double f(double x) const {
			double x2 = x*x;
			return 0.125*lambda*( x2*x2 - 6.0*x2 + 5.0 ) + 0.25*mu * (x2-1.0)*(x2-1.0);
		}
		double g(double x) const { return 0.25 * lambda * ( x*x - 1.0 ); }
		double h(double x) const { return compress_term(kappa,x); }
		double df(double x) const {
			double x2 = x*x;
			return 0.125*lambda*(4.0*x2*x - 12.0*x) + mu*x*(x2-1.0);
		}
		double dg(double x) const { return 0.5*lambda*x; }
		double dh(double x) const { return d_compress_term(kappa,x); }
	};

	class CoRotated : public Spline {
	public:
		CoRotated( double mu_, double lambda_, double kappa_ ) :
			mu(mu_), lambda(lambda_), kappa(kappa_) {}
		const double mu, lambda, kappa;
		double f(double x) const { return 0.5*lambda*(x*x - 6.0*x + 5.0) + mu*(x-1.0)*(x-1.0); }
		double g(double x) const { return lambda * (x - 1.0); }
		double h(double x) const { return compress_term(kappa,x); }
		double df(double x) const { return 0.5*lambda*(2.0*x - 6.0) + 2.0*mu*(x-1.0); }
		double dg(double x) const { (void)(x); return lambda; }
		double dh(double x) const { return d_compress_term(kappa,x); }
	};

} // end namespace xu

} // end namespace admm

#endif





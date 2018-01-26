// Copyright (c) 2016, University of Minnesota
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

#ifndef ADMM_SPRINGENERGYTERM_H
#define ADMM_SPRINGENERGYTERM_H 1

#include "EnergyTerm.hpp"

namespace admm {

//
//	To call it a spring is a bit misleading. It's an infinitely hard
//	hard spring, with energy = inf when violated, zero otherwise.
//
class SpringPin : public EnergyTerm {
protected:
	typedef Eigen::Matrix<int,3,1> Vec3i;
	typedef Eigen::Matrix<double,3,1> Vec3;
	typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
	int idx; // constrained vertex
	Vec3 pin; // location of the pin
	bool active;
	double weight;

public:
	int get_dim() const { return 6; }
	double get_weight() const { return weight; }
	void set_pin( const Vec3 &p ){ pin = p; }
	void set_active( bool a ){ active = a; }

	SpringPin( int idx_, const Vec3 &pin_ ) : idx(idx_), pin(pin_), active(true) {
		// Because we usually use bulk mod of rubber for elastics,
		// We'll make a really strong rubber and use that for pin.
		admm::Lame lame = admm::Lame::rubber();
		weight = std::sqrt(lame.bulk_modulus()*2.0);
	}

	void get_reduction( std::vector< Eigen::Triplet<double> > &triplets ){
		const int col = 3*idx;
		triplets.emplace_back( 0, col+0, 1.0 );
		triplets.emplace_back( 1, col+1, 1.0 );
		triplets.emplace_back( 2, col+2, 1.0 );
	}

	void prox( VecX &zi ){ if( active ){ zi = pin; } }

	double energy( const VecX &F ){
		if( !active ){ return 0.0; }
		return weight*(F-pin).norm(); // More useful than 0 and inf
	}

	double gradient( const VecX &F, VecX &grad ){
		(void)(F); (void)(grad);
		throw std::runtime_error("**SpringPin Error: No gradient for hard constraint");
	}

}; // end class TriEnergyTerm

} // end namespace admm

#endif





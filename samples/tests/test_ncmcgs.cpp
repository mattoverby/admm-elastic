// Copyright (c) 2017 University of Minnesota
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


#include "MCL/Vec.hpp"
#include "MCL/XForm.hpp"
#include "NodalMultiColorGS.hpp"
#include "TetEnergyTerm.hpp"
#include <unsupported/Eigen/SparseExtra>
#include <iostream>

using namespace Eigen;
typedef SparseMatrix<double,RowMajor> SparseMat;
typedef Matrix<double,Dynamic,1> VecX;

int main(){

	// Not really the most appropriate matrix since ncmcgs is stride=3.
	// The following test should still work anyway
	std::stringstream ss;
	ss << ADMMELASTIC_ROOT_DIR << "/samples/tests/bcsstk11.mtx";
	srand(100);
	SparseMat A;
	Eigen::loadMarket(A, ss.str());
	VecX x0 = VecX::Random(A.rows())*10.0;
	VecX b = A*x0;

	int runs = 10;
	double err_last = std::numeric_limits<double>::max();
	for( int i=1; i<=runs; ++i ){
		VecX x = VecX::Zero(A.rows());

		admm::NodalMultiColorGS solver;
		solver.max_iters = i*100;
		solver.m_tol = 0.0; // run max iters
		solver.update_system(A);
		solver.solve(x,b);

		double err = (x-x0).norm();
		std::cout << "iters: " << solver.max_iters << ", err: " <<  err << std::endl;

		if( err > err_last ){
			std::cerr << "Failed to improve convergence with increased iterations" << std::endl;
			return EXIT_FAILURE;
		}
		err_last = err;
	}
	std::cout << "SUCCESS" << std::endl;
	return EXIT_SUCCESS;
}

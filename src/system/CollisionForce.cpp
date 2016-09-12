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

#include "CollisionForce.hpp"

using namespace admm;

using namespace Eigen;

//// PUBLIC METHODS ////
	
void CollisionForce::computeDi( int dof ) { 
	Eigen::SparseMatrix<double> newDi;
	newDi.resize(dof,dof);
	newDi.setIdentity();
	setDi( newDi );
}
		
void CollisionForce::update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const { 
	
	// Computing Di * x + ui
	int Di_rows = getDi()->rows();
	Eigen::VectorXd Dix = Dx.segment( global_idx, Di_rows );
	Eigen::VectorXd ui = u.segment( global_idx, Di_rows );
	Eigen::VectorXd DixPlusUi = Dix + ui;
	Eigen::VectorXd zi(DixPlusUi.size());

	handleCollisions(zi,DixPlusUi);  // perturb zi as needed to handle collisions
	//zi = DixPlusUi;

	ui += ( Dix - zi );
	u.segment( global_idx, Di_rows ) = ui;
	z.segment( global_idx, Di_rows ) = zi;
}




//// PRIVATE METHODS ////

void CollisionForce::handleCollisions(Eigen::VectorXd &zi, const Eigen::VectorXd& collFreePositions) const{
	
	zi = collFreePositions;

	// Nested omp threads
#pragma omp parallel for
	for(int i = 0; i < zi.size(); i += 3){
		Eigen::Vector3d point(zi[i],zi[i+1],zi[i+2]);

		// Triple nested??? no...
		//#pragma omp parallel for
		for(int j = 0; j < collisionShapes.size(); j++){
			if( collisionShapes[j] -> isColliding(point) ){
					
				point = collisionShapes[j] -> projectOut(point);
				zi[i] = point[0];
				zi[i+1] = point[1];
				zi[i+2] = point[2];
			} 
		}	
	}
}


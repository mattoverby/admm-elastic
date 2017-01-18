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

#ifndef ADMM_TETFORCE_H
#define ADMM_TETFORCE_H 1

#include "Force.hpp"

namespace admm {

//
//	Helper Functions
//
namespace helper {
	static inline void add_block( Eigen::SparseMatrix<double> &mat, int blockRows, int blockCols, int r, int c, const Eigen::MatrixXd &block ){
		using namespace Eigen; for( int i=0; i<blockRows; ++i ){ for( int j=0; j<blockCols; ++j ){ mat.insert( r+i, c+j ) = block(i,j); } }
	}
	static inline void init_tet_force( int *idx, const Eigen::VectorXd &x, double &volume, Eigen::Matrix<double,4,3> &B, Eigen::Matrix<double,3,3> &edges_inv ){
		using namespace Eigen;

		// Edge matrix
		Matrix<double,3,3> edges;
		Vector3d v0( x[ idx[0]*3 ], x[ idx[0]*3+1 ], x[ idx[0]*3+2 ] );
		Vector3d v1( x[ idx[1]*3 ], x[ idx[1]*3+1 ], x[ idx[1]*3+2 ] );
		Vector3d v2( x[ idx[2]*3 ], x[ idx[2]*3+1 ], x[ idx[2]*3+2 ] );
		Vector3d v3( x[ idx[3]*3 ], x[ idx[3]*3+1 ], x[ idx[3]*3+2 ] );
		edges.col(0) = v1 - v0;
		edges.col(1) = v2 - v0;
		edges.col(2) = v3 - v0;
		edges_inv = edges.inverse();

		// Xg is beginning state		
		Matrix<double,3,3> Xg = edges;

		// D matrix is I dunno
		Matrix<double,4,3> D;
		D(0,0) = -1; D(0,1) = -1; D(0,2) = -1;
		D(1,0) =  1; D(1,1) =  0; D(1,2) =  0;
		D(2,0) =  0; D(2,1) =  1; D(2,2) =  0;
		D(3,0) =  0; D(3,1) =  0; D(3,2) =  1;

		// B is used to create Ai
		B = D * Xg.inverse();

		// Volume is used for weight
		volume = fabs( (v0-v3).dot( (v1-v3).cross(v2-v3) ) ) / 6.0;
	}
	static inline void init_tet_Di( int *idx, const Eigen::Matrix<double,4,3> &B, const int dof, Eigen::SparseMatrix<double> &Di ){
		using namespace Eigen;

		SparseMatrix<double> newDi;

		// Using dense tempAi matrix because block function is nice				
		newDi.resize(9,dof); newDi.setZero();
		Matrix<double,3,4> Bt = B.transpose();

		// Columns of the Di we care about
		const int col0 = 3 * idx[0];
		const int col1 = 3 * idx[1];
		const int col2 = 3 * idx[2];
		const int col3 = 3 * idx[3];

		const int rows[3] = { 0, 3, 6 };
		const int cols[4] = { col0, col1, col2, col3 };

		for( int r=0; r<3; ++r ){
			for( int c=0; c<4; ++c ){
				MatrixXd block = Matrix<double,3,3>::Identity(3,3);
				block *= Bt(r,c);
				helper::add_block( newDi, 3, 3, rows[r], cols[c], block );
			}
		}

		Di = newDi;
	}

	// Projection, Singular Values, SVD's U, SVD's V transpose
	static inline void oriented_svd( const Eigen::Matrix3d &F, Eigen::Vector3d &S, Eigen::Matrix3d &U, Eigen::Matrix3d &Vt ){
		using namespace Eigen;

		JacobiSVD< Matrix3d > svd( F, ComputeFullU | ComputeFullV );
		S = svd.singularValues();
		U = svd.matrixU();
		Vt = svd.matrixV().transpose();
		Matrix3d J = Matrix3d::Identity(); J(2,2)=-1.0;

		// Check for inversion: U
		if( U.determinant() < 0.0 ){
			U = U * J;
			S[2] *= -1.0;
		}

		// Check for inversion: V
		if( Vt.determinant() < 0.0 ){
			Vt = J * Vt;
			S[2] *= -1.0;
		}

		#ifdef PDADMM_VERIFY
		// Verify. Note: asserts don't work in OpenMP for some reason, so I'm using exit instead.
		if( U.determinant() < 0.0 ){ printf("\n\n**Error: Bad U\n\n"); exit(0); }
		if( Vt.determinant() < 0.0 ){ printf("\n\n**Error: Bad V\n\n"); exit(0); }
		#endif

	} // end oriented svd

} // end namespace helper

//
//	Linear Tet
//

class LinearTetStrain : public Force {
public:
	// If weight scale <= 0, the optimal-weight function is used (helper::tet_weight)
	LinearTetStrain( int idx0_, int idx1_, int idx2_, int idx3_, double stiffness_, double weight_scale_=1.f ) :
		stiffness(stiffness_), volume(0.0), weight_scale(weight_scale_)
		{ idx[0]=idx0_; idx[1]=idx1_; idx[2]=idx2_; idx[3]=idx3_; }

	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void computeDi( int dof );
	void update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;

//	double getEnergy( const Eigen::VectorXd& Dx, Eigen::VectorXd& grad ) const;

	int idx[4];
	double stiffness, volume, weight_scale;
	Eigen::Matrix<double,4,3> B;
	Eigen::Matrix3d edges_inv; // used for piola stress

}; // end class LinearTetStrain

//
//	Linear Tet Volume
//

class LinearTetVolume : public Force {
public:
	// If weight scale <= 0, the optimal-weight function is used (helper::tet_weight)
	LinearTetVolume( int idx0_, int idx1_, int idx2_, int idx3_, double rangeMin_, double rangeMax_, double stiffness_ ) :
		rangeMin(rangeMin_), rangeMax(rangeMax_), stiffness(stiffness_), volume(0.0)
		{ idx[0]=idx0_; idx[1]=idx1_; idx[2]=idx2_; idx[3]=idx3_; }

	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void computeDi( int dof );
	void update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;

	int idx[4];
	double rangeMin, rangeMax;
	double stiffness, volume;
	Eigen::Matrix<double,4,3> B;
	Eigen::Matrix3d edges_inv; // used for piola stress

}; // end class LinearTetVolume

//
//	Anisotropic Tet
//

class AnisotropicTet : public Force {
public:
	// If weight scale <= 0, the optimal-weight function is used (helper::tet_weight)
	AnisotropicTet( int idx0_, int idx1_, int idx2_, int idx3_, Eigen::Vector3d e_, double k_, double stiffness_ ) :
		e(e_), k(k_), stiffness(stiffness_), volume(0.0)
		{ idx[0]=idx0_; idx[1]=idx1_; idx[2]=idx2_; idx[3]=idx3_; eeT = e*e.transpose(); }

	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void computeDi( int dof );
	void update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;

	int idx[4];
	double stiffness, volume;
	double k;
	Eigen::Matrix<double,4,3> B;
	Eigen::Matrix3d edges_inv; // used for piola stress
	Eigen::Vector3d e;  // direction
	Eigen::Matrix3d eeT; // direction outer prod

}; // end class AnisotropicForce 

//
//	NeoHookean Hyper Elastic
//

// Proximal Operator for neohookean
class NHProx : public cppoptlib::Problem<double> {
public:
	NHProx(double scaleConst_, double mu_, double lambda_, double k_, double volume_ ) : scaleConst(scaleConst_), mu(mu_), lambda(lambda_), k(k_), volume(volume_) {}
	NHProx(double mu_, double lambda_, double k_, double volume_ ) : mu(mu_), lambda(lambda_), k(k_), volume(volume_) { scaleConst = 1.0; }
	void setSigma0( Eigen::Vector3d &Sigma_init_ ){ Sigma_init=Sigma_init_; }
	double energyDensity( const cppoptlib::Vector<double> &sigma) const;
	double value(const cppoptlib::Vector<double> &x);
	void gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad);
	void hessian(const cppoptlib::Vector<double> &x, cppoptlib::Matrix<double> &hess);
	Eigen::Vector3d Sigma_init;
	double scaleConst, mu, lambda, k, volume;

}; // end class nhprox

//
//	Saint Venant-Kirchhoff Hyper Elastic
//

// Proximal Operator for StVK
class StVKProx : public cppoptlib::Problem<double> {
public:
	StVKProx(double mu_, double lambda_, double k_, double volume_ ) : mu(mu_), lambda(lambda_), k(k_), volume(volume_) {}
	void setSigma0( Eigen::Vector3d &Sigma_init_ ){ Sigma_init=Sigma_init_; }
	double energyDensity( Eigen::Vector3d &Sigma ) const;
	double value(const cppoptlib::Vector<double> &x);
	void gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad);
	Eigen::Vector3d Sigma_init;
	double mu, lambda, k, volume;
	static inline double ddot( Eigen::Vector3d &a, Eigen::Vector3d &b ) { return (a*b.transpose()).trace(); }
	static inline double v3trace( const Eigen::Vector3d &v ) { return v[0]+v[1]+v[2]; }
}; // end class StVKProx

//
//	Hyper Elastic Tet Forces
//
//	Type is either:
//		nh	(0)
//		stvk	(1)
//
//	It's probably better to be able to pass in a proximal operator as an argument instead of the elastic
//	type for extensibility. However, this is easier to do for now.
//
class HyperElasticTet : public Force {
public:
	HyperElasticTet( int idx0_, int idx1_, int idx2_, int idx3_, double mu_, double lambda_, int max_iterations, std::string type_ ) :
		mu(mu_), lambda(lambda_) {
		idx[0]=idx0_; idx[1]=idx1_; idx[2]=idx2_; idx[3]=idx3_;
		type = 0; if( type_=="stvk" || type_=="1" ){ type=1; }

		// Set up the local solver
		solver = std::unique_ptr< cppoptlib::ISolver<double, 1> >( new cppoptlib::lbfgssolver<double> );
		solver->settings_.maxIter = max_iterations;
		solver->settings_.gradTol = 1e-8;
		last_prox_result.resize(3);
		last_prox_result.fill(1.0);
	}

	void initialize( const Eigen::VectorXd &x, const Eigen::VectorXd &v, const Eigen::VectorXd &masses, const double timestep );
	void computeDi( int dof );
	void update( double dt, const Eigen::VectorXd &Dx, Eigen::VectorXd &u, Eigen::VectorXd &z ) const;

	std::shared_ptr<NHProx> nhprox;
	std::shared_ptr<StVKProx> stvkprox;
	std::unique_ptr< cppoptlib::ISolver<double, 1> > solver;

	int idx[4];
	int type;
	double mu, lambda, volume;
	Eigen::Matrix<double,4,3> B;
	Eigen::Matrix3d edges_inv; // used for piola stress
	mutable cppoptlib::Vector<double> last_prox_result;

}; // end class HyperElastic


} // end namespace admm

#endif





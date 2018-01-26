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
#include "Solver.hpp"
#include "TetEnergyTerm.hpp"


using namespace mcl;
using namespace Eigen;
typedef Eigen::SparseMatrix<double> SparseMat;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;

static bool near( double a, double b, double eps=1e-12 ){ return std::abs(a-b) < eps; }
static double volume( const VecX &x ){
	Matrix<double,3,3> edges;
	edges.col(0) = x.segment<3>(3) - x.segment<3>(0);
	edges.col(1) = x.segment<3>(6) - x.segment<3>(0);
	edges.col(2) = x.segment<3>(9) - x.segment<3>(0);
	return (edges).determinant()/6.f;
}

class SingleTet {
public:
	std::vector< Vec3d > verts;
	Vec4i tet;
	SparseMat D;
	std::shared_ptr<admm::EnergyTerm> t;
	bool init( const admm::Lame lame = admm::Lame::rubber() );
	double volume();
	VecX x();
};

//
//	Check the energy of the lin tet
//
bool test_energy(){

	admm::Lame lame;
	lame.mu = 0;
	lame.lambda = 1;

	if( !near( lame.bulk_modulus(), 1.0 ) ){
		std::cerr << "Bad bulk modulus: " << lame.bulk_modulus() << std::endl;
		return false;
	}

	SingleTet tet;
	if( !tet.init(lame) ){
		std::cerr << "Failed tet init" << std::endl;
		return false;
	}

	// PD-weight (k) is computed as bulk mod * volume
	// and admm weight (w) is sqrt(k)
	double w = tet.t->get_weight();
	if( !near( lame.bulk_modulus()*tet.volume(), w*w ) ){
		std::cerr << "Did you change the weight function?" << std::endl;
		return false;
	}

	// At rest, energy should be zero
	double energy = tet.t->energy( tet.D, tet.x() );
	if( !near(energy,0.0) ){
		std::cerr << "Energy not zero at rest" << std::endl;
		return false;
	}

	// Rotate: no change in energy
	tet.init(lame);
	mcl::XForm<double> rotate = mcl::xform::make_rot( 45.0, Vec3d(1,1,1) );
	for( int i=0; i<(int)tet.verts.size(); ++i ){ tet.verts[i] = rotate * tet.verts[i]; }
	energy = tet.t->energy( tet.D, tet.x() );
	if( !near(energy,0.0) ){
		std::cerr << "Energy not zero after rotation" << std::endl;
		return false;
	}

	// Stretch
	tet.init(lame);
	mcl::XForm<double> scale = mcl::xform::make_scale( 2.0,2.0,2.0 );
	for( int i=0; i<(int)tet.verts.size(); ++i ){ tet.verts[i] = scale * tet.verts[i]; }
	energy = tet.t->energy( tet.D, tet.x() );
	if( !near(energy,0.25) ){
		std::cerr << "Energy not correct after deformation" << std::endl;
		return false;
	}

	// Strech new k with twice the stiffness.
	// because energy = (k/2) || Dx-p ||^2
	// and k = lambda + (2/3)m.
	lame.lambda = 2.123;
	tet.init(lame);
	for( int i=0; i<(int)tet.verts.size(); ++i ){ tet.verts[i] = scale * tet.verts[i]; }
	double prev_energy = energy;
	energy = tet.t->energy( tet.D, tet.x() );
	if( !near(energy,prev_energy*lame.lambda) || energy <= 0.0 ){ // should be twice the energy
		std::cerr << "Energy not correct after deformation" << std::endl;
		return false;
	}

	// After projecting it to the minimizng Dx (zi), the energy should be minimized
	tet.init(lame);
	VecX z = VecX::Random(tet.D.rows());
	VecX u = VecX::Zero(z.rows());
	VecX randX = VecX::Random(tet.x().rows());
	VecX Dx = tet.D*tet.x();
	tet.t->update( tet.D, tet.x(), z, u );

	// The ADMM constraint is W(Dx-z)=0
	double c_err = tet.t->get_weight() * (Dx - z).norm();
	if( !near(c_err,0.0) ){
		std::cerr << "Prox doesn't satisfy constraint" << std::endl;
		return false;
	}

	// Now lets check the deform grad with a simple scale
	tet.init(lame);
	scale = mcl::xform::make_scale( 3.1,4.2,5.3 );
	for( int i=0; i<(int)tet.verts.size(); ++i ){ tet.verts[i] = scale * tet.verts[i]; }
	Dx = tet.D*tet.x();
	Matrix<double,3,3> F = Map<Matrix<double,3,3> >(Dx.data());
	for( int r=0; r<3; ++r ){
		for( int c=0; c<3; ++c ){
			double v = F(r,c);
			if(r==c){
				if( !near(scale(r,c),v) ){
					std::cerr << "Bad deform grad (" << r << ", " << c << "): \n" << F << std::endl;
					return false;
				}
			} else {
				if( !near(0.0,v) ){
					std::cerr << "Bad deform grad (" << r << ", " << c << "): \n" << F << std::endl;
					return false;
				}
			}
		}
	}

	return true;
}

//
//	Test that the number of solver iterations
//	does not change result
//
bool test_solver_iters(){

	SingleTet tet;
	admm::Lame lame(500000,0.25); // a bit softer than real rubber
	if( !tet.init(lame) ){ return false; }

	admm::Solver solver;
	admm::Solver::Settings settings;
	settings.gravity = 0;
	settings.verbose = 0;
	settings.timestep_s = 1.f/24.f;
	settings.linsolver = 0; // LDLT
	solver.energyterms.push_back( tet.t );

	// Add vertex data:
	int n_verts = tet.verts.size();
	solver.m_x.resize( n_verts*3 );
	solver.m_masses = Eigen::VectorXd::Ones( n_verts*3 );
	for( int i=0; i<n_verts; ++i ){
		solver.m_x.segment<3>(i*3) = tet.verts[i].cast<double>();
	}

	Eigen::VectorXd init_x = solver.m_x;
	double last_error = -1;
	int start_iters = 5;
	double true_x = 52.2321; // rounded!

	for( int i=start_iters; i<100; ++i ){

		int idx = 9; // fourth tet
		settings.admm_iters = i;
		solver.m_x = init_x; // reset
		if( !solver.initialize(settings) ){ return false; }

		// Move out z and step
		solver.m_x.segment<3>(idx) = Vec3d(200,0,0);
		solver.step();

		// Should converge toward 52.2321
		double new_x = solver.m_x.segment<3>(idx)[0];

		// After 20 iters, we should be there
		if( i > 20 ){
			if( !near(true_x,new_x,1e-4) ){
				std::cerr << "Did not converge with iters (" << i << ")" << std::endl;
				std::cerr << "true: " << true_x << ", result: " << new_x << std::endl;
				return false;
			}
		}

		// Otherwise, we should at least be closer to the solution
		// than the previous run with less iterations.
		else if( last_error >= 1e-8 ) {
			double curr_error = (true_x-new_x)*(true_x-new_x);
			if( curr_error > last_error ){
				std::cerr << "Problem converging with increased iterations (" << i << ")" << std::endl;
				std::cerr << "Last err: " << last_error << ", curr err: " << curr_error << std::endl;
				return false;
			}
		}

		last_error = (true_x-new_x)*(true_x-new_x);
	}

	return true;
}

//
//	Tests that the solver is able to
//	resolve inversions with different num iters
//
bool test_inversion(){

	admm::Lame softlame;
	softlame.mu = 100;
	softlame.lambda = 100;

	SingleTet tet;
	if( !tet.init(softlame) ){ return false; }

	admm::Solver solver;
	admm::Solver::Settings settings;
	settings.gravity = 0;
	settings.verbose = 0;
	settings.timestep_s = 0.7;
	settings.linsolver = 0; // LDLT
	solver.energyterms.push_back( tet.t );

	// Add vertex data:
	solver.m_x = tet.x();
	solver.m_masses = Eigen::VectorXd::Ones( solver.m_x.rows() );
	Eigen::VectorXd init_x = solver.m_x;

	Vec3d last_x(0,0,0);
	double target_v = tet.volume();
	int start_iters = 10;

	for( int i=start_iters; i<100; ++i ){

		settings.admm_iters = i;
		solver.m_x = init_x; // reset
		if( !solver.initialize(settings) ){ return false; }

		if( !near(volume(solver.m_x), target_v ) ){
			std::cerr << "Error in init" << std::endl;
			return false;	
		}

		// Invert the tet
		solver.m_x.segment<3>(0) = Vec3d(1,1,1);
		if( volume(solver.m_x) > 0 ){
			std::cerr << "Didn't invert the tet" << std::endl;
			return false;
		}

		// Run several time steps.
		// The reason for multiple steps is that it may not have the correct
		// volume after one step from the momentum of fixing the inversion.
		// This goes away if you increase the stiffness.
		for( int j=0; j<10; ++j ){
			solver.step();
		}

		// Should converge to 0.75, 0.75, 0.75
		Vec3d curr_x = solver.m_x.segment<3>(0);

		// Check new volume
		double new_v = volume(solver.m_x);
		if( new_v <= 0.0 ){
			std::cerr << "Invert test: Did not fix inversion" << std::endl;
			return false;
		}

		double eps = 1e-6;

		if( !near(target_v,new_v,eps) ){
			std::cerr << "Iters: " << settings.admm_iters << std::endl;
			std::cerr << "Invert test: Diff solution with num admm iters (volume)" << std::endl;
			std::cerr << "Target v: " << target_v << ", new v: " << new_v << std::endl;
			return false;
		}

		if( settings.admm_iters > start_iters ){
			double diff = (last_x-curr_x).norm();
			if( !near(diff,0.f,eps) ){
				std::cerr << "Iters: " << settings.admm_iters << std::endl;
				std::cerr << "Invert test: Diff solution with num admm iters (position)" << std::endl;
				std::cerr << "Diff: " << diff << std::endl;
				return false;
			}
		}
		last_x = curr_x;
	}

	

	return true;

} // end test stretch


double SingleTet::volume(){
	Matrix<double,3,3> edges;
	edges.col(0) = verts[ tet[1] ] - verts[ tet[0] ];
	edges.col(1) = verts[ tet[2] ] - verts[ tet[0] ];
	edges.col(2) = verts[ tet[3] ] - verts[ tet[0] ];
	return (edges).determinant()/6.f;
}

VecX SingleTet::x(){
	VecX x0 = VecX::Zero(12);
	x0.segment<3>(0) = verts[0];
	x0.segment<3>(1*3) = verts[1];
	x0.segment<3>(2*3) = verts[2];
	x0.segment<3>(3*3) = verts[3];
	return x0;
}

bool SingleTet::init( const admm::Lame lame ){


	tet = Vec4i(0,1,2,3);
	verts = {
		Vec3d(0,0,0),
		Vec3d(0,1,0),
		Vec3d(0,0,1),
		Vec3d(1,0,0)
	};

	double V = volume();
	if( V <= 0.0 ){
		std::cerr << "Bad tet volume: " << V << std::endl;
		return false;
	}

	t = std::make_shared<admm::TetEnergyTerm>( admm::TetEnergyTerm(tet, verts, lame) );

	if( t->get_weight() <= 0 ){
		std::cerr << "Bad weight: " << t->get_weight() << std::endl;
		return false;
	}

	std::vector< Eigen::Triplet<double> > triplets;
	std::vector<double> weights;
	t->get_reduction(triplets,weights);

	if( weights.size() != 9 ){
		std::cerr << "Bad num weights: " << weights.size() << std::endl;
		return false;
	}

	if( triplets.size() != 36 ){
		std::cerr << "Bad num triplets: " << triplets.size() << std::endl;
		return false;
	}

	for( int i=0; i<(int)triplets.size(); ++i ){
		if( triplets[i].row() > 9 ){
			std::cerr << "Bad trip row: " << triplets[i].row() << std::endl;
			return false;
		}
		if( triplets[i].col() > 12 ){
			std::cerr << "Bad trip col: " << triplets[i].col() << std::endl;
			return false;
		}
	}

	D.resize(9,12);
	D.setFromTriplets(triplets.begin(),triplets.end());

	return true;
}

int main(){
	srand(100);
	bool success = true;
	success &= test_energy();
	success &= test_solver_iters();
	success &= test_inversion();
	if( !success ){
		std::cerr << "\n**FAILURE**\n" << std::endl;
		return EXIT_FAILURE;
	}
	std::cout << "SUCCESS" << std::endl;
	return EXIT_SUCCESS;
}

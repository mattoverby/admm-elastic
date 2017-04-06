// Copyright (c) 2016 University of Minnesota
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


#ifndef COLLISION_SPHERE_HPP
#define COLLISION_SPHERE_HPP

#include "CollisionShape.hpp"

namespace admm {

class CollisionSphere : public CollisionShape {
	
	public:
	
		CollisionSphere(Eigen::Vector3d shapeCenter, double sphRadius);
		
		double isColliding(Eigen::Vector3d pos) const;
		Eigen::Vector3d projectOut(const Eigen::Vector3d currPos) const;
		
		double radius;
		
	private:
	
};

CollisionSphere::CollisionSphere(Eigen::Vector3d shapeCenter, double sphRadius)
	: CollisionShape(shapeCenter) , radius(sphRadius) {
}

double CollisionSphere::isColliding(Eigen::Vector3d pos) const{
	
	double distance = radius - (pos - center).norm();
	return distance;
//	return distance <= radius;
	
}


Eigen::Vector3d CollisionSphere::projectOut(const Eigen::Vector3d currPos) const{
	
	Eigen::Vector3d disp = currPos - center;
	Eigen::Vector3d dir = disp / disp.norm();
	
	return center + radius * dir;
}

} // end of namespace admm

#endif

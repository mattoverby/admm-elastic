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

#ifndef COLLISION_CYLINDER_HPP
#define COLLISION_CYLINDER_HPP

#include "CollisionShape.hpp"

namespace admm {

class CollisionCylinder : public CollisionShape {
	
	public:
	
		CollisionCylinder(Eigen::Vector3d shapeCenter, Eigen::Vector3d cylScale, double cylRadius);
	
		double isColliding(Eigen::Vector3d pos) const;
		Eigen::Vector3d projectOut(const Eigen::Vector3d currPos) const;
	
		double radius;
		double length;
	
	private:
	
	
};


// NOTE: for now, we assume the cylinder's axis is parallel to the ground plane,
//       AND that the cylinder's central axis is parallel with the z direction

CollisionCylinder::CollisionCylinder(Eigen::Vector3d shapeCenter, Eigen::Vector3d cylScale, double cylRadius)
	: CollisionShape(Eigen::Vector3d(shapeCenter[0],shapeCenter[1],0)) , radius(cylRadius) {
}


double CollisionCylinder::isColliding(Eigen::Vector3d pos) const {
	Eigen::Vector3d posXY(pos[0],pos[1],0);
	double distance = radius - (posXY - center).norm();
	return distance;	
}


Eigen::Vector3d CollisionCylinder::projectOut(const Eigen::Vector3d currPos) const {
	Eigen::Vector3d posXY(currPos[0],currPos[1],0);
	Eigen::Vector3d disp = posXY - center;
	Eigen::Vector3d dir = disp / disp.norm();
	return center + (radius) * dir + Eigen::Vector3d(0,0,currPos[2]);
}

} // end of namespace admm

#endif

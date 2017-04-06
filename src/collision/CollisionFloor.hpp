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

#ifndef COLLISION_FLOOR_HPP
#define COLLISION_FLOOR_HPP

#include "CollisionShape.hpp"

namespace admm {

class CollisionFloor : public CollisionShape {
	
	public:
	
		CollisionFloor(Eigen::Vector3d shapeCenter);
		
		double isColliding(Eigen::Vector3d pos) const;
		Eigen::Vector3d projectOut(const Eigen::Vector3d currPos) const;
		
		double radius;
		
	private:
	
};


CollisionFloor::CollisionFloor(Eigen::Vector3d shapeCenter)
	: CollisionShape(shapeCenter) {
}

double CollisionFloor::isColliding(Eigen::Vector3d pos) const{
	return center[1]-pos[1];
//	return pos[1] <= center[1];
}


Eigen::Vector3d CollisionFloor::projectOut(const Eigen::Vector3d currPos) const{
	return Eigen::Vector3d(currPos[0],center[1],currPos[2]);
}



} // end of namespace admm

#endif


// @HEADER
// ************************************************************************
//
//               Rapid Optimization Library (ROL) Package
//                 Copyright (2014) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact lead developers:
//              Drew Kouri   (dpkouri@sandia.gov) and
//              Denis Ridzal (dridzal@sandia.gov)
//
// ************************************************************************
// @HEADER


#ifndef ROL_DOUGLASRACHFORDPROJECTION_H
#define ROL_DOUGLASRACHFORDPROJECTION_H

#include "ROL_PolyhedralProjection.hpp"
#include "ROL_ParameterList.hpp"

namespace ROL {

template<typename Real>
class DouglasRachfordProjection : public PolyhedralProjection<Real> {
private:
  int dim_;
  Ptr<Vector<Real>> tmp_, p_, q_, y_, z_;
  Real b_, cdot_;

  Real DEFAULT_atol_, DEFAULT_rtol_;
  int DEFAULT_maxit_, DEFAULT_verbosity_;
  Real DEFAULT_alpha1_, DEFAULT_gamma_, DEFAULT_t0_;

  Real atol_, rtol_;
  int maxit_, verbosity_;
  Real alpha1_, alpha2_, gamma_, t0_;

  using PolyhedralProjection<Real>::bnd_;
  using PolyhedralProjection<Real>::con_;
  using PolyhedralProjection<Real>::xprim_;
  using PolyhedralProjection<Real>::xdual_;
  using PolyhedralProjection<Real>::mul_;
  using PolyhedralProjection<Real>::res_;

public:

  DouglasRachfordProjection(const Vector<Real>               &xprim,
                    const Vector<Real>               &xdual,
                    const Ptr<BoundConstraint<Real>> &bnd,
                    const Ptr<Constraint<Real>>      &con,
                    const Vector<Real>               &mul,
                    const Vector<Real>               &res);

  DouglasRachfordProjection(const Vector<Real>               &xprim,
                    const Vector<Real>               &xdual,
                    const Ptr<BoundConstraint<Real>> &bnd,
                    const Ptr<Constraint<Real>>      &con,
                    const Vector<Real>               &mul,
                    const Vector<Real>               &res,
                    ParameterList                    &list);

  void project(Vector<Real> &x, std::ostream &stream = std::cout) override;

private:

  Real residual_1d(const Vector<Real> &x) const;

  void residual_nd(Vector<Real> &r, const Vector<Real> &y) const;

  void project_bnd(Vector<Real> &x, const Vector<Real> &y) const;

  void project_con(Vector<Real> &x, const Vector<Real> &y) const;

  void project_DouglasRachford(Vector<Real> &x, std::ostream &stream = std::cout) const;

}; // class DouglasRachfordProjection

} // namespace ROL

#include "ROL_DouglasRachfordProjection_Def.hpp"

#endif

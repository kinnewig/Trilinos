// @HEADER
// ************************************************************************
//
//        Piro: Strategy package for embedded analysis capabilitites
//                  Copyright (2010) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Andy Salinger (agsalin@sandia.gov), Sandia
// National Laboratories.
//
// ************************************************************************
// @HEADER

#include "Piro_LOCASolver_Def.hpp"

namespace Piro {

// Explicit template instantiations
// LOCA currently only supports Scalar = double

template class LOCASolver<double>;

template Teuchos::RCP<LOCASolver<double> > observedLocaSolver(
    const Teuchos::RCP<Teuchos::ParameterList> &appParams,
    const Teuchos::RCP<Thyra::ModelEvaluator<double> > &model,
    const Teuchos::RCP<Thyra::ModelEvaluator<double> > &adjointModel,
    const Teuchos::RCP<Piro::ObserverBase<double> > &observer);

} // namespace Piro


#include "Teuchos_OrdinalTraits.hpp"
#include "Teuchos_toString.hpp"

Piro::Detail::ModelEvaluatorParamName::ModelEvaluatorParamName(
    const Teuchos::RCP<const Teuchos::Array<std::string> > &p_names) :
  p_names_(p_names)
{
  if (Teuchos::is_null(p_names_)) {
    type_ = Default;
  } else if (p_names_->size() == Teuchos::OrdinalTraits<Teuchos_Ordinal>::one()) {
    type_ = OneShared;
  } else {
    type_ = FullList;
  }
}

std::string
Piro::Detail::ModelEvaluatorParamName::operator()(Teuchos_Ordinal k) const
{
  switch (type_) {
    case Default:
      return "Parameter " + Teuchos::toString(k);
    case OneShared:
      return p_names_->front();
    case FullList:
      return (*p_names_)[k];
  }

  TEUCHOS_TEST_FOR_EXCEPT(true);
  return std::string();
}

#ifndef _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DEF_HPP
#define _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DEF_HPP

#include <FROSch_OptimizedSchwarzOperator_decl.hpp>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

      template <class SC, class LO, class GO, class NO>
    OptimizedSchwarzOperator<SC, LO, GO, NO>::OptimizedSchwarzOperator(GraphPtr graph)
    : SchwarzOperator<SC,LO,GO,NO>(graph->getRowMap()->getComm())
    {
        Graph_ = graph;
    }



    template <class SC, class LO, class GO, class NO>
    int OptimizedSchwarzOperator<SC, LO, GO, NO>::initialize()
    {
        // TODO
    }



    template <class SC, class LO, class GO, class NO>
    typename OptimizedSchwarzOperator<SC,LO,GO,NO>::GraphPtr OptimizedSchwarzOperator<SC, LO, GO, NO>::getOverlappingGraph()
    {
        return Graph_; 
    }

    

    template <class SC, class LO, class GO, class NO>
    void OptimizedSchwarzOperator<SC, LO, GO, NO>::setNeumanMatrix(ConstXMatrixPtr matrix)
    {
        // TODO
    }
    
    
    
    template <class SC, class LO, class GO, class NO>
    void OptimizedSchwarzOperator<SC, LO, GO, NO>::setMassMatrix(ConstXMatrixPtr matrix)
    {
        // TODO
    }



    template <class SC, class LO, class GO, class NO>
    int OptimizedSchwarzOperator<SC, LO, GO, NO>::compute()
    {
        // TODO
    }



    template <class SC, class LO, class GO, class NO>
    void OptimizedSchwarzOperator<SC, LO, GO, NO>::apply(
        const XMultiVector &x,
        XMultiVector &y,
        bool usePreconditionerOnly,
        ETransp mode,
        SC alpha,
        SC beta) const
    {
        // TODO
    }



    template <class SC, class LO, class GO, class NO>
    void OptimizedSchwarzOperator<SC, LO, GO, NO>::describe(
        FancyOStream &out,
        const EVerbosityLevel verbLevel) const
    {
       // TODO
    }



    template <class SC, class LO, class GO, class NO>
    string OptimizedSchwarzOperator<SC, LO, GO, NO>::description() const
    {
      return "Optimized Schwarz Method";
    }


} // namespace FROSch

#endif // _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DEF_HPP

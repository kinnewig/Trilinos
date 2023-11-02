#ifndef _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DECL_HPP
#define _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DECL_HPP

#include <FROSch_SchwarzOperator_def.hpp>

namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class OptimizedSchwarzOperator : public SchwarzOperator<SC,LO,GO,NO> {

    protected:

        using XMultiVector                      = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVector;

        using XMatrix                           = typename SchwarzOperator<SC,LO,GO,NO>::XMatrix;
        using XMatrixPtr                        = typename SchwarzOperator<SC,LO,GO,NO>::XMatrixPtr;
        using ConstXMatrixPtr                   = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMatrixPtr;

        using XCrsGraph                         = typename SchwarzOperator<SC,LO,GO,NO>::XCrsGraph;
        using GraphPtr                          = typename SchwarzOperator<SC,LO,GO,NO>::GraphPtr;
        using ConstXCrsGraphPtr                 = typename SchwarzOperator<SC,LO,GO,NO>::ConstXCrsGraphPtr;

        using ParameterListPtr                  = typename SchwarzOperator<SC,LO,GO,NO>::ParameterListPtr;
        using CommPtr                           = typename SchwarzOperator<SC,LO,GO,NO>::CommPtr;

    public:
    
        /**
         *  Constructor from a Teuchos::RCP<Xpetra::CrsGraph<SC,LO,GO,NO>>
         *  where the graph contains the dual representation of the matrix
         */
        OptimizedSchwarzOperator(GraphPtr graph);
    
        int initialize();

        GraphPtr getOverlappingGraph();

        void setNeumanMatrix(ConstXMatrixPtr);

        void setMassMatrix(ConstXMatrixPtr);

        int compute();


        void apply(const XMultiVector &x,
                   XMultiVector &y,
                   bool usePreconditionerOnly,
                   ETransp mode=NO_TRANS,
                   SC alpha=ScalarTraits<SC>::one(),
                   SC beta=ScalarTraits<SC>::zero()) const;

        void describe(FancyOStream &out,
                      const EVerbosityLevel verbLevel=Describable::verbLevel_default) const;

        string description() const;

    protected:
      
        /**
         *  Store a Teuchos::RCP<Xpetra::CrsGraph<SC,LO,GO,NO>> 
         *  which contains the dual representation of the matrix
         */  
        GraphPtr Graph_;

    }; //class OptimizedSchwarzOperator

} // namespace FROSch

#endif // _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DECL_HPP

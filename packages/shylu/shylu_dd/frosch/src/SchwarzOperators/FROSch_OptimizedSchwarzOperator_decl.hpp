#ifndef _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DECL_HPP
#define _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DECL_HPP

#include <FROSch_SchwarzOperator_def.hpp>

namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    /**
     *
     * @tparam SC
     * @tparam LO
     * @tparam GO
     * @tparam NO
     */
    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class OptimizedSchwarzOperator : public SchwarzOperator<SC,LO,GO,NO> {

    protected:

        using XLongLongMultiVector              = Xpetra::MultiVector<long long,LO,GO,NO>; // TODO?
        using XLongLongMultiVectorPtr           = RCP<XLongLongMultiVector>;
        using XMultiVector                      = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVector;
        using XMultiVectorPtr                   = typename SchwarzOperator<SC,LO,GO,NO>::XMultiVectorPtr;

        using XMatrix                           = typename SchwarzOperator<SC,LO,GO,NO>::XMatrix;
        using XMatrixPtr                        = typename SchwarzOperator<SC,LO,GO,NO>::XMatrixPtr;
        using ConstXMatrixPtr                   = typename SchwarzOperator<SC,LO,GO,NO>::ConstXMatrixPtr;

        using XCrsGraph                         = typename SchwarzOperator<SC,LO,GO,NO>::XCrsGraph;
        using GraphPtr                          = typename SchwarzOperator<SC,LO,GO,NO>::GraphPtr;
        using ConstXCrsGraphPtr                 = typename SchwarzOperator<SC,LO,GO,NO>::ConstXCrsGraphPtr;

        using XMap                              = typename SchwarzOperator<SC,LO,GO,NO>::XMap;
        using XMapPtr                           = typename SchwarzOperator<SC,LO,GO,NO>::XMapPtr;

        using ParameterListPtr                  = typename SchwarzOperator<SC,LO,GO,NO>::ParameterListPtr;
        using CommPtr                           = typename SchwarzOperator<SC,LO,GO,NO>::CommPtr;

    public:

        /**
         *  Constructor from the dual graph and the local to global vertex map
         */
        OptimizedSchwarzOperator(GraphPtr graph, XMapPtr map);

        int initialize();

        /**
         * TODO: improve this description
         * @param cell_list Description of the cell data.
         * @param vertex_list Description of the vertex list.
         */
        int initialize(
            XLongLongMultiVectorPtr &cell_data,
            XMultiVectorPtr &vertex_list,
            long long numGlobalVertices);

        GraphPtr getOverlappingGraph();

        /**
         *  Set a value to the NeumannMatrix.
         */
        void setNeumannMatrix(XMatrixPtr);

        /**
         *  Set a value RobinMatrix.
         */
        void setRobinMatrix(XMatrixPtr);

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
         *  Store a RCP<Xpetra::CrsGraph<SC,LO,GO,NO>>
         *  which contains the dual graph, i.e.
         *  if there is an entry in (row i, column j)
         *  element i and element j are neighbors.
         */
        GraphPtr DualGraph_;

        /**
         * The local to global map for the vertices.
         * local_index = VertexMap_.getLocalElement(global_index);
         */
         XMapPtr VertexMap_;

        /**
         *  The Neumann matrix, where each rank only
         *  holds the Neumann matrix, that belongs to its
         *  subdomain.
         */
         XMatrixPtr NeumannMatrix_;

         /**
          * The Robin matrix describes the Robin boundary.
          * Therefore this matrix has only entries, which
          * are correlating to boundary elements.
          * Each rank holds only the Robin matrix, that belongs
          * to its subdomain.
          */
          XMatrixPtr RobinMatrix_;

    }; //class OptimizedSchwarzOperator

} // namespace FROSch

#endif // _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DECL_HPP

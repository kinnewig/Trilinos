#ifndef _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DEF_HPP
#define _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DEF_HPP

#include <FROSch_OptimizedSchwarzOperator_decl.hpp>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC, class LO, class GO, class NO>
    OptimizedSchwarzOperator<SC, LO, GO, NO>::OptimizedSchwarzOperator(GraphPtr graph, XMapPtr map)
    : SchwarzOperator<SC,LO,GO,NO>(graph->getRowMap()->getComm())
    , DualGraph_(graph)
    , VertexMap_(map)
    {}


  template <class SC, class LO, class GO, class NO>
  int
  OptimizedSchwarzOperator<SC, LO, GO, NO>::initialize()
  {}



    template <class SC, class LO, class GO, class NO>
    int
    OptimizedSchwarzOperator<SC, LO, GO, NO>::initialize(
        XLongLongMultiVectorPtr &cell_data,
        XMultiVectorPtr         &vertex_list,
        long long                numGlobalVertices)
    {

      // TODO: können wir das hier verallgemeinern, damit auch andere Overlap größen
      //       möglich sind, als nur overlap = 2?
      RCP<const Xpetra::CrsGraph<LO,GO,NO>> OverlappingDualGraph;
      RCP<const Xpetra::Map<LO,GO,NO>> OverlappingMap = DualGraph_->getColMap();
      RCP<const Xpetra::Map<LO,GO,NO>> OverlappingMap2;
      FROSch::ExtendOverlapByOneLayer<LO,GO,NO>(DualGraph_, DualGraph_->getColMap(), OverlappingDualGraph, OverlappingMap);
      FROSch::ExtendOverlapByOneLayer<LO,GO,NO>(OverlappingDualGraph, OverlappingMap, OverlappingDualGraph, OverlappingMap2);

      // get information about the system
      long long verticesPerCell = cell_data->getNumVectors();

      // Communicate the cell data
      XLongLongMultiVectorPtr                cellDataOverlapping = Xpetra::MultiVectorFactory<long long,LO,GO,NO>::Build(OverlappingMap, verticesPerCell);
      Teuchos::RCP<Xpetra::Import<LO,GO,NO>> cellImporter        = Xpetra::ImportFactory<LO,GO,NO>::Build(DualGraph_->getRowMap(), OverlappingMap);
      cellDataOverlapping->doImport(*cell_data,*cellImporter,Xpetra::INSERT);

      long long  numLocalElemtents = cellDataOverlapping->getLocalLength();

      // extract the new local to global vertex map from cell_data_overlapping
      // TODO: use kokkos::views
      Teuchos::Array<long long> array(numLocalElemtents * verticesPerCell);
      for (unsigned int i = 0; i < verticesPerCell; ++i ) {
        auto data = cellDataOverlapping->getData(i);
        for (unsigned int j = 0; j < numLocalElemtents; ++j) {
          array[(j * verticesPerCell) + i] = data[j];
        }
      }

      // Since, we just added all global vertex indices from the cell_data,
      // there are now a lot of duplicates. Let's remove those.
      FROSch::sortunique(array);

      XMapPtr overlappingVertexMap = Xpetra::MapFactory<LO,GO,NO>::Build(DualGraph_->getMap()->lib(), numGlobalVertices, array(), 0, DualGraph_->getMap()->getComm());

      // Communicate the vertex list
      XMultiVectorPtr vertexListOverlapping = Xpetra::MultiVectorFactory<double,LO,GO,NO>::Build(overlappingVertexMap,vertex_list->getNumVectors); //TODO
      RCP<Xpetra::Import<LO,GO,NO> > vertex_importer = Xpetra::ImportFactory<LO,GO,NO>::Build(VertexMap_, overlappingVertexMap);
      vertexListOverlapping->doImport(*vertex_list,*vertex_importer,Xpetra::INSERT);

      // use the overlappingVertexMap to replace the globalDofIndices with localDofIndices
      // in the cellDataOverlapping
      // TODO: use kokkos::views
      for (unsigned int i = 0; i < verticesPerCell; ++i ) {
        auto data = cellDataOverlapping->getDataNonConst(i);
        for (unsigned int j = 0; j < numLocalElemtents; ++j) {
          data[j] = overlappingVertexMap->getLocalElement(data[j]);
        }
      }

      cell_data = cellDataOverlapping;
      vertex_list = vertexListOverlapping;
    }



    template <class SC, class LO, class GO, class NO>
    typename OptimizedSchwarzOperator<SC,LO,GO,NO>::GraphPtr
    OptimizedSchwarzOperator<SC, LO, GO, NO>::getOverlappingGraph()
    {
        return DualGraph_;
    }



    template <class SC, class LO, class GO, class NO>
    void OptimizedSchwarzOperator<SC, LO, GO, NO>::setNeumannMatrix(XMatrixPtr matrix)
    {
      NeumannMatrix_ = matrix;
    }



    template <class SC, class LO, class GO, class NO>
    void OptimizedSchwarzOperator<SC, LO, GO, NO>::setRobinMatrix(XMatrixPtr matrix)
    {
      RobinMatrix_ = matrix;
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

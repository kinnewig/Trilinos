#ifndef _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DEF_HPP
#define _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DEF_HPP

#include <FROSch_OptimizedSchwarzOperator_decl.hpp>


namespace FROSch {

    using namespace std;
    using namespace Teuchos;
    using namespace Xpetra;

    template <class SC, class LO, class GO, class NO>
    OptimizedSchwarzOperator<SC, LO, GO, NO>::OptimizedSchwarzOperator(GraphPtr Graph,
                                                                       XMapPtr Map) :
    OverlappingOperator<SC,LO,GO,NO>(Graph->getRowMap()->getComm()),
    DualGraph_ (Graph),
    VertexMap_ (Map)
    {}


    template <class SC, class LO, class GO, class NO>
    int
    OptimizedSchwarzOperator<SC, LO, GO, NO>::initialize()
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"OptimizedSchwarzOperator::initialize");
        FROSCH_ASSERT(false,"ERROR: initialize() is not avaible OptimizedSchwarzOperator");
    }


    template <class SC, class LO, class GO, class NO>
    int
    OptimizedSchwarzOperator<SC, LO, GO, NO>::initialize(int overlap)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"OptimizedSchwarzOperator::initialize");

        if (this->Verbose_) {
            cout
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| "
            << left << setw(74) << "AlgebraicOverlappingOperator " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")"
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "========================================================================================="
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Adding layers strategy" << right
            << " | " << setw(41) << this->ParameterList_->get("Adding Layers Strategy","CrsGraph")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Combine mode in overlap" << right
            << " | " << setw(41) << this->ParameterList_->get("Combine Values in Overlap","Restricted")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Solver type" << right
            << " | " << setw(41) << this->ParameterList_->sublist("Solver").get("SolverType","Amesos2")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Solver" << right
            << " | " << setw(41) << this->ParameterList_->sublist("Solver").get("Solver","Klu")
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << "| " << left << setw(41) << "Reuse symbolic factorization" << right
            << " | " << setw(41) << this->ParameterList_->get("Reuse: Symbolic Factorization",true)
            << " |"
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << endl;
        }

        this->buildOverlappingGraph(overlap);

        this->IsInitialized_ = true;
        this->IsComputed_ = false;
        return 0; // RETURN VALUE!!!
    }

    template <class SC, class LO, class GO, class NO>
    int
    OptimizedSchwarzOperator<SC, LO, GO, NO>::communicateTriangulation(XLongLongMultiVectorPtr ElementList,
                                                                       XMultiVectorPtr         NodeList,
                                                                       XLongLongMultiVectorPtr &ElementListOverlapping,
                                                                       XMultiVectorPtr         &NodeListOverlapping)
    {
        FROSCH_TIMER_START_LEVELID(initializeTime,"OptimizedSchwarzOperator::communicateTriangulation");

        // get information about the system
        long long nodesPerCell = ElementList->getNumVectors();

        // Communicate the cell data
        ElementListOverlapping = Xpetra::MultiVectorFactory<long long,LO,GO,NO>::Build(OverlappingMap, nodesPerCell);
        Teuchos::RCP<Xpetra::Import<LO,GO,NO>> elementImporter = Xpetra::ImportFactory<LO,GO,NO>::Build(DualGraph_->getRowMap(), this->OverlappingMap_);
        ElementListOverlapping->doImport(*ElementList,*elementImporter,Xpetra::INSERT);

        long long  numLocalElemtents = ElementListOverlapping->getLocalLength();

        // extract the new local to global vertex map from ElementList_overlapping
        // TODO: use kokkos::views
        Teuchos::Array<long long> array(numLocalElemtents * nodesPerCell);
        for (unsigned int i = 0; i < nodesPerCell; ++i ) {
            auto data = ElementListOverlapping->getData(i);
            for (unsigned int j = 0; j < numLocalElemtents; ++j) {
              array[(j * nodesPerCell) + i] = data[j];
            }
        }

        // Since, we just added all global vertex indices from the ElementList,
        // there are now a lot of duplicates. Let's remove those.
        FROSch::sortunique(array);

        XMapPtr overlappingNodeMap = Xpetra::MapFactory<LO,GO,NO>::Build(DualGraph_->getMap()->lib(), DualGraph_->getMap()->getMaxGlobalIndex()+1, array(), 0, DualGraph_->getMap()->getComm());

        // Communicate the vertex list
        NodeListOverlapping = Xpetra::MultiVectorFactory<double,LO,GO,NO>::Build(overlappingNodeMap,NodeList->getNumVectors); //TODO
        RCP<Xpetra::Import<LO,GO,NO> > vertex_importer = Xpetra::ImportFactory<LO,GO,NO>::Build(NodeList->getMap(), overlappingNodeMap);
        NodeListOverlapping->doImport(*NodeList,*vertex_importer,Xpetra::INSERT);

        // use the overlappingNodeMap to replace the globalDofIndices with localDofIndices
        // in the ElementListOverlapping
        // TODO: use kokkos::views
        for (unsigned int i = 0; i < nodesPerCell; ++i ) {
            auto data = ElementListOverlapping->getDataNonConst(i);
            for (unsigned int j = 0; j < numLocalElemtents; ++j) {
                data[j] = overlappingNodeMap->getLocalElement(data[j]);
            }
        }
    }


    template <class SC, class LO, class GO, class NO>
    typename OptimizedSchwarzOperator<SC,LO,GO,NO>::GraphPtr
    OptimizedSchwarzOperator<SC, LO, GO, NO>::getOverlappingGraph()
    {
        return DualGraph_;
    }



    template <class SC, class LO, class GO, class NO>
    void OptimizedSchwarzOperator<SC, LO, GO, NO>::setNeumannMatrix(ConstXMatrixPtr NeumannMatrix)
    {
      NeumannMatrix_ = NeumannMatrix;
    }


    template <class SC, class LO, class GO, class NO>
    void OptimizedSchwarzOperator<SC, LO, GO, NO>::setRobinMatrix(ConstXMatrixPtr RobinMatrix)
    {
      RobinMatrix_ = RobinMatrix;
    }


    template <class SC, class LO, class GO, class NO>
    int OptimizedSchwarzOperator<SC, LO, GO, NO>::compute()
    {
        FROSCH_TIMER_START_LEVELID(computeTime,"OptimizedSchwarzOperator::compute");

        // Compute
        XMatrix overlappingMatrix = MatrixFactory<SC,LO,GO,NO>::BuildCopy(NeumannMatrix_);
        TwoMatrixAdd (RobinMatrix_,false,this->ParameterList_->get("Robin BC: alpha",1.0),overlappingMatrix);

        computeOverlappingOperator();
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


    template <class SC,class LO,class GO,class NO>
    int OptimizedSchwarzOperator<SC,LO,GO,NO>::buildOverlappingGraph(int overlap)
    {
        FROSCH_DETAILTIMER_START_LEVELID(buildOverlappingMatricesTime,"AlgebraicOverlappingOperator::buildOverlappingMatrices");
        // ====================================================================================
        // AH 08/09/2019: This is just temporary. Implement this properly in all the classes
        Verbosity verbosity = All;
        if (!this->ParameterList_->get("Verbosity","All").compare("None")) {
            verbosity = None;
        } else if (!this->ParameterList_->get("Verbosity","All").compare("All")) {
            verbosity = All;
        } else {
            FROSCH_ASSERT(false,"FROSch::AlgebraicOverlappingOperator: Specify a valid verbosity level.");
        }
        // ====================================================================================

        this->OverlappingMap_ = DualGraph_->getColMap();
        this->OverlappingGraph_ = DualGraph_->K_;

        GO global = 0, sum = 0;
        LO local,minVal,maxVal;
        SC avg;
        if (verbosity==All) {
            FROSCH_DETAILTIMER_START_LEVELID(printStatisticsTime,"print statistics");

            global = this->OverlappingMap_->getMaxAllGlobalIndex();
            if (this->OverlappingMap_->lib()==UseEpetra || this->OverlappingMap_->getGlobalNumElements()>0) {
                global += 1;
            }

            local = (LO) max((LO) this->OverlappingMap_->getLocalNumElements(),(LO) 0);
            reduceAll(*this->MpiComm_,REDUCE_SUM,GO(local),ptr(&sum));
            avg = max(sum/double(this->MpiComm_->getSize()),0.0);
            reduceAll(*this->MpiComm_,REDUCE_MIN,local,ptr(&minVal));
            reduceAll(*this->MpiComm_,REDUCE_MAX,local,ptr(&maxVal));

            if (this->Verbose_) {
                cout
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| "
                << left << setw(74) << "> Overlapping Subdomains Statistics " << right << setw(8) << "(Level " << setw(2) << this->LevelID_ << ")"
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "========================================================================================="
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << " " << right
                << " | " << setw(10) << "total"
                << " | " << setw(10) << "avg"
                << " | " << setw(10) << "min"
                << " | " << setw(10) << "max"
                << " | " << setw(10) << "global sum"
                << " |"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << setw(89) << "-----------------------------------------------------------------------------------------"
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << setw(20) << "Layer 0" << right
                << " | " << setw(10) << global
                << " | " << setw(10) << setprecision(5) << avg
                << " | " << setw(10) << minVal
                << " | " << setw(10) << maxVal
                << " | " << setw(10) << sum
                << " |";
            }
        }
    }

    // Adding Layers of Elements to the overlapping subdomains
    ConstXCrsGraphPtr overlappingGraph = this->OverlappingMatrix_->getCrsGraph();
    for (int i=1; i<overlap; i++) {
            ExtendOverlapByOneLayer(this->OverlappingGraph_,this->OverlappingMap_,this->OverlappingGraph_,this->OverlappingMap_);
        }
        if (verbosity==All) {
            FROSCH_DETAILTIMER_START_LEVELID(printStatisticsTime,"print statistics");
            local = (LO) max((LO) this->OverlappingMap_->getLocalNumElements(),(LO) 0);
            reduceAll(*this->MpiComm_,REDUCE_SUM,GO(local),ptr(&sum));
            avg = max(sum/double(this->MpiComm_->getSize()),0.0);
            reduceAll(*this->MpiComm_,REDUCE_MIN,local,ptr(&minVal));
            reduceAll(*this->MpiComm_,REDUCE_MAX,local,ptr(&maxVal));

            if (this->Verbose_) {
                cout
                << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
                << "| " << left << "Layer " << setw(14) << i+1 << right
                << " | " << setw(10) << global
                << " | " << setw(10) << setprecision(5) << avg
                << " | " << setw(10) << minVal
                << " | " << setw(10) << maxVal
                << " | " << setw(10) << sum
                << " |";
            }
        }
    }

    if (verbosity==All) {
        FROSCH_DETAILTIMER_START_LEVELID(printStatisticsTime,"print statistics");
        if (this->Verbose_) {
            cout
            << "\n" << setw(FROSCH_OUTPUT_INDENT) << " "
            << setw(89) << "-----------------------------------------------------------------------------------------"
            << endl;
        }
    }

    // AH 08/28/2019 TODO: It seems that ExtendOverlapByOneLayer_Old is currently the fastest method because the map is sorted. This seems to be better for the direct solver. (At least Klu)
    if (this->ParameterList_->get("Sort Overlapping Map",true)) {
        this->OverlappingMap_ = SortMapByGlobalIndex(this->OverlappingMap_);
    }

} // namespace FROSch

#endif // _FROSCH_OPTIMIZEDSCHWARZOPERATOR_DEF_HPP

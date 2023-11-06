//@HEADER
// ************************************************************************
//
//               ShyLU: Hybrid preconditioner package
//                 Copyright 2012 Sandia Corporation
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
// Questions? Contact Alexander Heinlein (a.heinlein@tudelft.nl)
//
// ************************************************************************
//@HEADER

#include <ShyLU_DDFROSch_config.h>

#include <mpi.h>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>

// Xpetra include
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <Xpetra_MapFactory.hpp>

#include <FROSch_Tools_def.hpp>


using UN    = unsigned;
using SC    = double;
using LO    = int;
using GO    = FROSch::DefaultGlobalOrdinal;
using NO    = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType;

using namespace std;
using namespace Teuchos;
using namespace Xpetra;
using namespace FROSch;

using Teuchos::tuple;

int main(int argc, char *argv[])
{
    oblackholestream blackhole;
    GlobalMPISession mpiSession(&argc,&argv,&blackhole);

    RCP<const Comm<int> > CommWorld = DefaultPlatform::getDefaultPlatform().getComm();

    CommandLineProcessor My_CLP;

    RCP<FancyOStream> out = VerboseObjectBase::getDefaultOStream();

    bool useepetra = false;
    My_CLP.setOption("USEEPETRA","USETPETRA",&useepetra,"Use Epetra infrastructure for the linear algebra.");

    My_CLP.recogniseAllOptions(true);
    My_CLP.throwExceptions(false);
    CommandLineProcessor::EParseCommandLineReturn parseReturn = My_CLP.parse(argc,argv);
    if (parseReturn == CommandLineProcessor::PARSE_HELP_PRINTED) {
        return(EXIT_SUCCESS);
    }

    CommWorld->barrier();
    RCP<StackedTimer> stackedTimer = rcp(new StackedTimer("Overlap Test"));
    TimeMonitor::setStackedTimer(stackedTimer);

    FROSCH_ASSERT(CommWorld->getSize()==4,"Currently, the test is only implemented for the case of 4 MPI ranks.");
    // int N = (int) (pow(CommWorld->getSize(),1/2.) + 100*numeric_limits<double>::epsilon()); // 1/H

    UnderlyingLib XpetraLib = UseTpetra;
    if (useepetra) {
        XpetraLib = UseEpetra;
    } else {
        XpetraLib = UseTpetra;
    }

    RCP<FancyOStream> fancy = fancyOStream(rcpFromRef(cout));

    // +----+----++----+----+
    // | 12 | 13 || 14 | 15 |
    // +----+----++----+----+
    // |  8 |  9 || 10 | 11 |
    // +====+====++====+====+
    // |  4 |  5 ||  6 |  7 |
    // +----+----++----+----+
    // |  0 |  1 ||  2 |  3 |
    // +----+----++----+----+

    RCP<Map<LO,GO,NO> > RowMap;
    vector<GO> GlobalIndices;
    switch (CommWorld->getRank()) {
        case 0:
                GlobalIndices = {0,1,4,5};      RowMap = MapFactory<LO,GO,NO>::Build(XpetraLib,16,arrayViewFromVector<GO>(GlobalIndices),0,CommWorld);
                break;
        case 1:
                GlobalIndices = {2,3,6,7};      RowMap = MapFactory<LO,GO,NO>::Build(XpetraLib,16,arrayViewFromVector<GO>(GlobalIndices),0,CommWorld);
                break;
        case 2:
                GlobalIndices = {8,9,12,13};    RowMap = MapFactory<LO,GO,NO>::Build(XpetraLib,16,arrayViewFromVector<GO>(GlobalIndices),0,CommWorld);
                break;
        case 3:
                GlobalIndices = {10,11,14,15};  RowMap = MapFactory<LO,GO,NO>::Build(XpetraLib,16,arrayViewFromVector<GO>(GlobalIndices),0,CommWorld);
                break;
        default:
            FROSCH_ASSERT(false,"This cannot happen.");
            break;
    }

    RCP<CrsGraph<LO,GO,NO> > DualGraph = CrsGraphFactory<LO,GO,NO>::Build(RowMap,4);
    vector<GO> Columns;

    switch (CommWorld->getRank()) {
        case 0:
            Columns = {1,4};        DualGraph->insertGlobalIndices(0,arrayViewFromVector<GO>(Columns));
            Columns = {0,2,5};      DualGraph->insertGlobalIndices(1,arrayViewFromVector<GO>(Columns));
            Columns = {0,5,8};      DualGraph->insertGlobalIndices(4,arrayViewFromVector<GO>(Columns));
            Columns = {1,4,6,9};    DualGraph->insertGlobalIndices(5,arrayViewFromVector<GO>(Columns));
            break;
        case 1:
            Columns = {1,3,6};      DualGraph->insertGlobalIndices(2,arrayViewFromVector<GO>(Columns));
            Columns = {2,7};        DualGraph->insertGlobalIndices(3,arrayViewFromVector<GO>(Columns));
            Columns = {2,5,7,10};   DualGraph->insertGlobalIndices(6,arrayViewFromVector<GO>(Columns));
            Columns = {3,6,11};     DualGraph->insertGlobalIndices(7,arrayViewFromVector<GO>(Columns));
            break;
        case 2:
            Columns = {4,9,12};     DualGraph->insertGlobalIndices(8,arrayViewFromVector<GO>(Columns));
            Columns = {5,8,10,13};  DualGraph->insertGlobalIndices(9,arrayViewFromVector<GO>(Columns));
            Columns = {8,13};       DualGraph->insertGlobalIndices(12,arrayViewFromVector<GO>(Columns));
            Columns = {9,12,14};    DualGraph->insertGlobalIndices(13,arrayViewFromVector<GO>(Columns));
            break;
        case 3:
            Columns = {6,9,11,14};  DualGraph->insertGlobalIndices(10,arrayViewFromVector<GO>(Columns));
            Columns = {7,10,15};    DualGraph->insertGlobalIndices(11,arrayViewFromVector<GO>(Columns));
            Columns = {10,13,15};   DualGraph->insertGlobalIndices(14,arrayViewFromVector<GO>(Columns));
            Columns = {11,14};      DualGraph->insertGlobalIndices(15,arrayViewFromVector<GO>(Columns));
            break;
        default:
            FROSCH_ASSERT(false,"This cannot happen.");
            break;
    }
    DualGraph->fillComplete(RowMap,RowMap);

    DualGraph->describe(*fancy,VERB_EXTREME);

    RCP<const CrsGraph<LO,GO,NO> > BoundaryElementsGraph;
    FindBoundaryElements<LO,GO,NO>(DualGraph,BoundaryElementsGraph);

    BoundaryElementsGraph->describe(*fancy,VERB_EXTREME);

    RCP<const CrsGraph<LO,GO,NO> > OverlappingDualGraph;
    RCP<const Map<LO,GO,NO> > OverlappingMap = DualGraph->getColMap();
    RCP<const Map<LO,GO,NO> > OverlappingMap2;
    ExtendOverlapByOneLayer<LO,GO,NO>(DualGraph,OverlappingMap,OverlappingDualGraph,OverlappingMap2);

    OverlappingDualGraph->describe(*fancy,VERB_EXTREME);

    RCP<const CrsGraph<LO,GO,NO> > OverlappingBoundaryElementsGraph;
    FindBoundaryElements<LO,GO,NO>(OverlappingDualGraph,OverlappingBoundaryElementsGraph);

    OverlappingBoundaryElementsGraph->describe(*fancy,VERB_EXTREME);

    // Define MultiVectors
    int NumColumns = 1;
    RCP<MultiVector<double,LO,GO,NO> > DoubleVectorNO = MultiVectorFactory<double,LO,GO,NO>::Build(RowMap,NumColumns);
    DoubleVectorNO->putScalar(double(CommWorld->getRank()));
    RCP<MultiVector<int,LO,GO,NO> > IntVectorNO = MultiVectorFactory<int,LO,GO,NO>::Build(RowMap,NumColumns);
    IntVectorNO->putScalar(CommWorld->getRank());

    DoubleVectorNO->describe(*fancy,VERB_EXTREME);
    IntVectorNO->describe(*fancy,VERB_EXTREME);

    // Communicate MultiVectors
    RCP<MultiVector<double,LO,GO,NO> > DoubleVectorO = MultiVectorFactory<double,LO,GO,NO>::Build(OverlappingMap,NumColumns);
    RCP<MultiVector<int,LO,GO,NO> > IntVectorO = MultiVectorFactory<int,LO,GO,NO>::Build(OverlappingMap,NumColumns);

    RCP<Import<LO,GO,NO> > importer = ImportFactory<LO,GO,NO>::Build(RowMap,OverlappingMap);
    DoubleVectorO->doImport(*DoubleVectorNO,*importer,INSERT);
    IntVectorO->doImport(*IntVectorNO,*importer,INSERT);

    DoubleVectorO->describe(*fancy,VERB_EXTREME);
    IntVectorO->describe(*fancy,VERB_EXTREME);

    // Show Timer
    CommWorld->barrier();
    stackedTimer->stop("Overlap Test");
    StackedTimer::OutputOptions options;
    options.output_fraction = options.output_histogram = options.output_minmax = true;
    stackedTimer->report(*out,CommWorld,options);

    return(EXIT_SUCCESS);

}

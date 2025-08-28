// Xpetra include
#include <Xpetra_ConfigDefs.hpp>
#include <Xpetra_DefaultPlatform.hpp>
#include <Xpetra_Map.hpp>
#include <Xpetra_CrsGraph.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_CrsGraphFactory.hpp>
#include <Xpetra_Parameters.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>

// FROSch include
#include <ShyLU_DDFROSch_config.h>
#include <FROSch_Tools_def.hpp>
#include <FROSch_GeometricOneLevelPreconditioner_decl.hpp>
#include <FROSch_GeometricOneLevelPreconditioner_def.hpp>

// Teuchos include
#include <Teuchos_RCP.hpp>
#include <Teuchos_GlobalMPISession.hpp> 
#include <Teuchos_DefaultComm.hpp> 
#include <Teuchos_OrdinalTraits.hpp> 
#include <Teuchos_Array.hpp> 

// Belos
#include <BelosLinearProblem.hpp>
#include <BelosSolverFactory.hpp>
#include <BelosXpetraAdapter.hpp>

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>


using UN    = unsigned;
using SC    = double;
using LO    = int;
using GO    = FROSch::DefaultGlobalOrdinal;
using NO    = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType;

using namespace std;
using namespace Teuchos;
using namespace Xpetra;
using namespace FROSch;
using namespace Belos;

typedef MultiVector<SC, LO, GO, NO> multivector_type;
typedef Belos::OperatorT<multivector_type> operatort_type;
typedef Belos::LinearProblem<SC, multivector_type, operatort_type> linear_problem_type;
typedef Belos::SolverFactory<SC, multivector_type, operatort_type> solverfactory_type;
typedef Belos::SolverManager<SC, multivector_type, operatort_type> solver_type;
typedef XpetraOp<SC, LO , GO, NO> xpetraop_type;

bool
contains(Teuchos::Array<GO> array, GO value) {
  for(GO i : array)
    if (i == value)
      return true;
  return false;
}


Array<GO>
extractIndexList(RCP<MultiVector<GO, LO, GO, NO>>  mv) {
  const size_t m = mv->getLocalLength();

  Array<GO> indexList(m);
  for (unsigned int j = 0; j < m; ++j)
    indexList[j] = j;

  {
    auto data = mv->getData(0);
    std::sort(indexList.begin(),
              indexList.end(),
              [data](unsigned int a, unsigned int b) {
                return data[a] < data[b];
              });
  }

  return indexList;;
}



template <typename Number>
Array<Array<Number>>
extractRcpMultiVector(RCP<MultiVector<Number, LO, GO, NO>>  mv) {
  const size_t n = mv->getNumVectors();
  const size_t m = mv->getLocalLength();

  Array<Array<Number>> list(m, Array<Number>(n));
  for (unsigned int i = 0; i < n; ++i)
    {
      auto data = mv->getData(i);

      for (unsigned int j = 0; j < m; ++j)
        list[j][i] = data[j];
    }

  return list;
}



RCP<CrsMatrix<SC, LO, GO, NO>> 
assembleMatrix(
    RCP<Map<LO, GO, NO>>              locallyOwnedDofs, 
    RCP<Map<LO, GO, NO>>              locallyRelevantDofs,
    RCP<MultiVector<GO, LO, GO, NO>>  cellVector,
    Array<GO>                         dofsOnBoundary
){
  unsigned int nGlobalDofs = locallyOwnedDofs->getGlobalNumElements();

  RCP<CrsMatrix<SC, LO, GO, NO>> returnMatrix = 
    CrsMatrixFactory<SC, LO, GO, NO>::Build(locallyOwnedDofs, nGlobalDofs);

  Array<Array<double>> localStiffnessMatrix = 
    {Array<double>{ 4.0/6.0, -1.0/6.0, -1.0/6.0, -2.0/6.0}, 
     Array<double>{-1.0/6.0,  4.0/6.0, -1.0/3.0, -1.0/6.0},
     Array<double>{-1.0/6.0, -1.0/3.0,  4.0/6.0, -1.0/6.0},
     Array<double>{-2.0/6.0, -1.0/6.0, -1.0/6.0,  4.0/6.0}};

  Array<Array<GO>> cellDataArray = extractRcpMultiVector(cellVector);

  // for simplicity we first create a dense matrix and create a sparse matrix out of that later.
  Array<Array<double>> fullMatrix(nGlobalDofs, Array<double>(nGlobalDofs));
  for(size_t cell = 0; cell < cellVector->getLocalLength(); ++cell)
    for(GO i = 0; i < 4; ++i)
      for(GO j = 0; j < 4; ++j)
          fullMatrix[cellDataArray[cell][i]][cellDataArray[cell][j]] += localStiffnessMatrix[i][j];

  // Apply boundary conditions
  for(GO i : dofsOnBoundary) 
    for(GO j = 0; j < nGlobalDofs; ++j) 
      if (i != j) {
        fullMatrix[i][j] = 0;
        fullMatrix[j][i] = 0;
      }

  // Create the distributed sparse matrix 
  for (auto globalRow : locallyRelevantDofs->getLocalElementList()) {
    Array<GO> cols;
    Array<SC> vals;
    for (unsigned int i = 0; i < nGlobalDofs; ++i) {
      if (std::abs(fullMatrix[globalRow][i]) > 1e-6) {
        cols.push_back(i);
        vals.push_back(fullMatrix[globalRow][i]);
      }
    }
    returnMatrix->insertGlobalValues(globalRow, cols, vals);
  }
  returnMatrix->fillComplete();

  return returnMatrix;
}



RCP<MultiVector<SC, LO, GO, NO>>
assembleRHS(
    RCP<Map<LO, GO, NO>>              locallyOwnedDofs,
    Array<GO>                         dofsOnBoundary
) {
  RCP<MultiVector<SC, LO, GO, NO>> returnVector = MultiVectorFactory<SC, LO, GO, NO>::Build(locallyOwnedDofs, 1);
  returnVector->putScalar( 1.0/16.0 );

  for (auto gid : dofsOnBoundary) 
    if (locallyOwnedDofs->isNodeGlobalElement(gid))
      returnVector->replaceLocalValue(locallyOwnedDofs->getLocalElement(gid), 0, 0.0); 

  return returnVector;
}



RCP<CrsMatrix<SC, LO, GO, NO>> 
assembleInterfaceMatrix(
    RCP<Map<LO, GO, NO>>              locallyOwnedDofs,
    RCP<MultiVector<GO, LO, GO, NO>>  cellVector,
    Array<GO>                         dofsOnInterface
  ){

  unsigned int nGlobalDofs = locallyOwnedDofs->getGlobalNumElements();

  RCP<CrsMatrix<SC, LO, GO, NO>> returnMatrix = 
    CrsMatrixFactory<SC, LO, GO, NO>::Build(locallyOwnedDofs, nGlobalDofs);

  Array<Array<GO>> cellDataArray = extractRcpMultiVector(cellVector);

  // for simplicity we first create a dense matrix and create a sparse matrix out of that later.
  Array<Array<double>> fullMatrix(nGlobalDofs, Array<double>(nGlobalDofs));
  for(size_t cell = 0; cell < cellVector->getLocalLength(); ++cell)
    for(GO i = 0; i < 4; ++i) {
      if(!contains(dofsOnInterface, cellDataArray[cell][i]))
        continue;
      for(GO j = 0; j < 4; ++j)
        if(contains(dofsOnInterface, cellDataArray[cell][j]))
          fullMatrix[cellDataArray[cell][i]][cellDataArray[cell][j]] += 0.0625;
    }

  // Create the distributed sparse matrix 
  for (auto globalRow : locallyOwnedDofs->getLocalElementList()) {
    Array<GO> cols;
    Array<SC> vals;
    for (unsigned int i = 0; i < nGlobalDofs; ++i) {
      if (std::abs(fullMatrix[globalRow][i]) > 1e-6) {
        cols.push_back(i);
        vals.push_back(fullMatrix[globalRow][i]);
      }
    }
    returnMatrix->insertGlobalValues(globalRow, cols, vals);
  }
  returnMatrix->fillComplete();

  return returnMatrix;
}



int 
main(int argc, char* argv[]) 
{
  oblackholestream blackhole;
  GlobalMPISession mpiSession(&argc,&argv,&blackhole);

  RCP<const Comm<int> > CommWorld = DefaultPlatform::getDefaultPlatform().getComm();
  RCP<const Comm<int> > CommSelf  = Teuchos::rcp(new MpiComm<int>(MPI_COMM_SELF));

  RCP<ParameterList> parameterList = getParametersFromXmlFile("ParameterList.xml");
  RCP<ParameterList> belosList     = sublist(parameterList,"Belos List");
  RCP<ParameterList> precList      = sublist(parameterList,"Preconditioner List");

  RCP<FancyOStream> out = VerboseObjectBase::getDefaultOStream();   

  int rank = CommWorld->getRank();


  /* We consider the following grid:
   *
   * +----+----+----+----+
   * | 12 | 13 | 14 | 15 |
   * +----+----+----+----+
   * |  8 |  9 | 10 | 11 |
   * +----+----+----+----+
   * 
   * +----+----+----+----+
   * |  4 |  5 |  6 |  7 | 
   * +----+----+----+----+
   * |  0 |  1 |  2 |  3 |
   * +----+----+----+----+
   *
   * Where the indices from 0 - 7 are stored on rank 0,
   * and the indices from 8 - 15 are stored on rank 1.
   *
   * With the dof enumeration
   *
   * 20---21---22---23---24
   * |    |    |    |    |
   * 15---16---17---18---19
   * |    |    |    |    |
   * 10---11---12---13---14
   * 
   * 10---11---12---13---14
   * |    |    |    |    | 
   * 2----3----5----7----9
   * |    |    |    |    |
   * 0----1----4----6----8
   * */

  // Geometric information:
  unsigned int dim               = 2;
  unsigned int vertices_per_cell = 4;
  GO           n_local_vertices  = 15;
  GO           nGlobalDofs       = 25;


  // ---------------------------------------------------------------------------------
  // Step 1: Locally owned index set

  // Remark: this would normally be provided by the finite element library
  Teuchos::RCP<Xpetra::Map<LO, GO, NO>> rowMap;
  {
    Array<GO> localIndices;

    if (rank == 0) 
      localIndices = Array<GO>{0, 1, 2, 3, 4, 5, 6, 7};

    else if (rank == 1)
      localIndices = Array<GO>{8, 9, 10, 11, 12, 13, 14, 15};

    rowMap = Xpetra::MapFactory<LO, GO, NO>::Build(
      Xpetra::UseTpetra, 16, localIndices, 0,  CommWorld);
  }


  // ---------------------------------------------------------------------------------
  // Step 2: Dual graph

  // create the connectivity map, i,e, the dual_graph
  // Remark: this would normally be provided by the finite element library
  RCP<CrsGraph<LO, GO, NO>> dualGraph = CrsGraphFactory<LO, GO, NO>::Build(rowMap, 4); 
  {
    Array<Array<GO>> neighbours;
    if (rank == 0) {
      neighbours = Array<Array<GO>>{
        {1, 4},
        {0, 2, 5},
        {1, 3, 6},
        {2, 7},
        {0, 5, 8},
        {1, 4, 6, 9},
        {2, 5, 7, 10},
        {3, 6, 11}
      };
    } else if (rank == 1) {
      neighbours = Array<Array<GO>>{
        {9, 12, 4},
        {8, 10, 13, 5},
        {9, 11, 14, 6},
        {10, 15, 7},
        {8, 13},
        {9, 12, 14},
        {10, 13, 15},
        {11, 14}
      };
    }

    for (GO i = 0; i < neighbours.size(); ++i) 
      dualGraph->insertGlobalIndices(rowMap->getGlobalElement(i), neighbours[i]);

    dualGraph->fillComplete();
  }


  // ---------------------------------------------------------------------------------
  // Step 3: Locally relevant and locally owned DoF list

  // Locally relevant DoF list:
  // Remark: this would normally be provided by the finite element library
  RCP<Map<LO, GO, NO>> locallyRelevantDofs;
  {
    Array<GO> indices(n_local_vertices);
    if (rank == 0) 
      for(unsigned int vertex_index = 0; vertex_index < n_local_vertices; ++vertex_index)
        indices[vertex_index] = vertex_index;
    else if (rank == 1)
      for(unsigned int vertex_index = 0; vertex_index < n_local_vertices; ++vertex_index)
        indices[vertex_index] = 10 + vertex_index;

    locallyRelevantDofs= Xpetra::MapFactory<LO, GO, NO>::Build(UseTpetra, 25, indices, 0, CommWorld);
  }

  // Locally owned DoF list:
  // Remark: this would normally be provided by the finite element library
  RCP<Map<LO, GO, NO>> locallyOwnedDofs;
  {
    Array<GO> indices;
    if (rank == 0) 
      for(unsigned int vertex_index = 0; vertex_index < 15; ++vertex_index)
        indices.push_back(vertex_index);
    else if (rank == 1)
      for(unsigned int vertex_index = 15; vertex_index < nGlobalDofs; ++vertex_index)
        indices.push_back(vertex_index);

    locallyOwnedDofs = Xpetra::MapFactory<LO, GO, NO>::Build(UseTpetra, 25, indices, 0, CommWorld);
  }



  // ---------------------------------------------------------------------------------
  // Step 4: Triangulation description

  // Node Data
  //   Stores a list of vertices present in the triangulation. 
  //   (We use that the dofs correspond to the vertices in this setting)
  RCP<MultiVector<double, LO, GO, NO>> nodesVector = 
    MultiVectorFactory<double, LO, GO, NO>::Build(locallyRelevantDofs, dim);
  {
    // Create an Array, such we can access its data
    Array<ArrayRCP<double>> nodesVectorData(dim);
    for (unsigned int i = 0; i < dim; ++i)
      nodesVectorData[i] = nodesVector->getDataNonConst(i);

    // Fill node_vector: 
    // (this part is normally provided by the FEM-Software)
    if (rank == 0)
      for(unsigned int vertex_index = 0; vertex_index < n_local_vertices; ++vertex_index) {
        nodesVectorData[0][vertex_index] = (vertex_index % 5) / 4.0;         // x-component of the node location
        nodesVectorData[1][vertex_index] = (vertex_index / 5) / 4.0;         // y-component of the node location
      }
    else if (rank == 1)
      for(unsigned int vertex_index = 0; vertex_index < n_local_vertices; ++vertex_index) {
        nodesVectorData[0][vertex_index] =  (vertex_index % 5) / 4.0;         // x-component of the node location
        nodesVectorData[1][vertex_index] = ((vertex_index / 5) / 4.0) + 0.5;  // y-component of the node location
      }
  }
   
  // Cell Data
  //   Store a description of each cell present in the triangulation.
  //   Each cell is described by it's vertices.
  RCP<MultiVector<GO, LO, GO, NO>> cellVector = 
    MultiVectorFactory<GO, LO, GO, NO>::Build(rowMap, vertices_per_cell);
  {
    // Create an Array, such we can access its data
    Array<ArrayRCP<GO>> cellVectorData(vertices_per_cell);
    for (unsigned int i = 0; i < vertices_per_cell; ++i)
      cellVectorData[i] = cellVector->getDataNonConst(i);

    // Prepare the information:
    // (this part is normally provided by the FEM-Software)
    Array<Array<GO>> cellDataArray;
    if (rank == 0) 
      cellDataArray = 
        Array<Array<GO>>{
          Array<GO>{0,1,2,3},   Array<GO>{1,4,3,5},   Array<GO>{4,6,5,7},   Array<GO>{6,8,7,9},
          Array<GO>{2,3,10,11}, Array<GO>{3,5,11,12}, Array<GO>{5,7,12,13}, Array<GO>{7,9,13,14}
        };
    else if (rank == 1)
      cellDataArray = 
        Array<Array<GO>>{
          Array<GO>{10,11,15,16}, Array<GO>{11,12,16,17}, Array<GO>{12,13,17,18}, Array<GO>{13,14,18,19},
          Array<GO>{15,16,20,21}, Array<GO>{16,17,21,22}, Array<GO>{17,18,22,23}, Array<GO>{18,19,23,24}
        };

    // copy the data into the cell_data_vector:
    for(unsigned int cell_index = 0; cell_index < 8; ++cell_index ) 
      for(unsigned int cell_vertex_index = 0; cell_vertex_index < vertices_per_cell; ++cell_vertex_index) 
        cellVectorData[cell_vertex_index][cell_index] = cellDataArray[cell_index][cell_vertex_index];
  }

  // Auxillary Data:
  //   Here we store the global cell index. What information is stored in the 
  //   auxillary vector depends on used FEM-Software.
  RCP<MultiVector<GO, LO, GO, NO>> auxillaryVector = 
    MultiVectorFactory<GO, LO, GO, NO>::Build(rowMap, 1);
  {
    ArrayRCP<GO> auxillaryVectorData = auxillaryVector->getDataNonConst(0);
    if (rank == 0) 
      for(unsigned int cell_index = 0; cell_index < 8; ++cell_index ) 
        auxillaryVectorData[cell_index] = cell_index;
    else if (rank == 1)
      for(unsigned int cell_index = 0; cell_index < 8; ++cell_index ) 
        auxillaryVectorData[cell_index] = cell_index + 8;
  }


  // ---------------------------------------------------------------------------------
  // Step 5: Assemble the linear system

  // Assemble system Matrix:
  // With the Node Data and the Cell Data we can assemble the system matrix
  // (this is normally done by the FEM-Software)
  RCP<CrsMatrix<SC, LO, GO, NO>> systemMatrix = 
    assembleMatrix(locallyOwnedDofs, 
                   locallyRelevantDofs, 
                   cellVector,
                   Array<GO>{0, 1, 4, 6, 8, 2, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24} /*dofs on boundary*/);

  // Assemble system RHS
  // (this is normally done by the FEM-Software)
  RCP<MultiVector<SC, LO, GO, NO>> systemRHS = 
    assembleRHS(locallyOwnedDofs, 
                   Array<GO>{0, 1, 4, 6, 8, 2, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24} /*dofs on boundary*/);

  // Debugging:
  //auto print_out = Teuchos::getFancyOStream (Teuchos::rcpFromRef(std::cout));
  //systemMatrix->describe(*print_out, Teuchos::VERB_EXTREME);


  // ---------------------------------------------------------------------------------
  // Step 6: Communicate between ranks
  
  /* The overlapping domains should look like this:
   * rank 1:
   * 15---16---17---18---19
   * |    |    |    |    |
   * 10---11---12---13---14
   * |    |    |    |    | 
   * 5----6----7----8----9
   * |    |    |    |    |
   * 0----1----2----3----4
   * 
   * rank 0:
   * 15---16---17---18---19
   * |    |    |    |    |
   * 10---11---12---13---14
   * |    |    |    |    | 
   * 2----3----5----7----9
   * |    |    |    |    |
   * 0----1----4----6----8
   */
  
  // convert to Xpetra::Matrix
  RCP<Matrix<double, LO, GO, NO>> k = Teuchos::rcp(new CrsMatrixWrap<double, LO, GO, NO>(systemMatrix));

  // Test the intialize function:
  RCP<GeometricOneLevelPreconditioner<double, LO, GO, NO>> geometricPreconditioner =
    rcp(new GeometricOneLevelPreconditioner<double, LO, GO, NO>(k.getConst(), dualGraph, precList));


  // Test the communication function:
  geometricPreconditioner->communicateOverlappingTriangulation(nodesVector,
                                                               cellVector,
                                                               auxillaryVector,
                                                               nodesVector,
                                                               cellVector,
                                                               auxillaryVector);


  // ---------------------------------------------------------------------------------
  // Step 7: Build the local system
 
  // Reorder the output to match the original cell enumeration
  { 
    Array<GO> indexList = extractIndexList(auxillaryVector);

    // create a deep copy:
    RCP<MultiVector<GO, LO, GO, NO>> cellVectorOriginal = 
      MultiVectorFactory<GO, LO, GO, NO>::Build(cellVector, Teuchos::Copy);

    Array<ArrayRCP<GO>>       cellVectorData(vertices_per_cell);
    Array<ArrayRCP<const GO>> cellVectorDataSource(vertices_per_cell);
    for (unsigned int i = 0; i < vertices_per_cell; ++i) {
      cellVectorData[i]       = cellVector->getDataNonConst(i);
      cellVectorDataSource[i] = cellVectorOriginal->getData(i);
    }

    for(unsigned int cell_index = 0; cell_index < cellVectorData[0].size(); ++cell_index ) 
      for(unsigned int cell_vertex_index = 0; cell_vertex_index < vertices_per_cell; ++cell_vertex_index) 
        cellVectorData[cell_vertex_index][cell_index] = cellVectorDataSource[cell_vertex_index][indexList[cell_index]];
  }

  // Debugging
  //auto print_out = Teuchos::getFancyOStream (Teuchos::rcpFromRef(std::cout));
  //cellVector->describe(*print_out, Teuchos::VERB_EXTREME);

  RCP<Map<LO,GO,NO>> localDofs;
  {
    Array<GO> indices(20);
    for(unsigned int vertex_index = 0; vertex_index < 20; ++vertex_index)
          indices[vertex_index] = vertex_index;

    localDofs= Xpetra::MapFactory<LO, GO, NO>::Build(UseTpetra, 20, indices, 0, CommSelf);
  }

  Array<GO> localDofsOnBoundary;
  Array<GO> localDofsOnInterface;
  if (rank == 0) {
    localDofsOnBoundary  = Array<GO>{0, 1, 2, 4, 6, 8, 9, 10, 14, 15, 16, 17, 18, 19};
    localDofsOnInterface = Array<GO>{15, 16, 17, 18, 19};
  } else if (rank == 1) {
    localDofsOnBoundary  = Array<GO>{0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 16, 17, 18, 19};
    localDofsOnInterface = Array<GO>{0, 1, 2, 3, 4};
  }


  // ---------------------------------------------------------------------------------
  // Step 8: Initialize the GeometricOneLevelPreconditioner
  
  // create the overlapping map:
  Array<GO> overlappingArray(20);
  if (rank == 0)
    for(GO i = 0; i < 20; ++i)
      overlappingArray[i] = i; 
  else if (rank == 1)
  {
    overlappingArray[0] = 2;
    overlappingArray[1] = 3;
    overlappingArray[2] = 5;
    overlappingArray[3] = 7;
    for(GO i = 9; i < 25; ++i)
      overlappingArray[i-9] = + i; 
  }
  Teuchos::RCP<Xpetra::Map<LO, GO, NO>> overlappingMap = 
    Xpetra::MapFactory<LO, GO, NO>::Build(Xpetra::UseTpetra, 20, overlappingArray, 0, CommWorld);

  geometricPreconditioner->initialize(overlappingMap);


  // ---------------------------------------------------------------------------------
  // Step 9: Assemble the local system

  RCP<CrsMatrix<SC, LO, GO, NO>> localNeumannMatrix = 
    assembleMatrix(localDofs, 
                   localDofs, 
                   cellVector,
                   localDofsOnBoundary);

  RCP<CrsMatrix<SC, LO, GO, NO>> localInterfaceMatrix = 
    assembleInterfaceMatrix(localDofs,
                            cellVector,
                            localDofsOnInterface);

  // Debugging
  //auto print_out = Teuchos::getFancyOStream (Teuchos::rcpFromRef(std::cout));
  //if (rank == 0)
  //  localInterfaceMatrix->describe(*print_out, Teuchos::VERB_EXTREME);


  // ---------------------------------------------------------------------------------
  // Step 10: Compute the preconditioner
  
  // convert to Xpetra::Matrix
  RCP<Matrix<SC, LO, GO, NO>> localNeumannMatrixConvert = Teuchos::rcp(new CrsMatrixWrap<SC, LO, GO, NO>(localNeumannMatrix));
  RCP<Matrix<SC, LO, GO, NO>> localInterfaceMatrixConvert = Teuchos::rcp(new CrsMatrixWrap<SC, LO, GO, NO>(localInterfaceMatrix));

  geometricPreconditioner->compute(localNeumannMatrixConvert, localInterfaceMatrixConvert);


  // ---------------------------------------------------------------------------------
  // Step 11: Solve
  
  RCP<MultiVector<SC, LO, GO, NO>> solution = MultiVectorFactory<SC, LO, GO, NO>::Build(locallyOwnedDofs, 1);
  
  // Convert the geometricPreconditioner to a belos preconditioner:
  RCP< Xpetra::Operator<SC, LO, GO, NO> > xpetraOperator = 
    rcp_dynamic_cast<Xpetra::Operator<SC, LO, GO, NO>>(geometricPreconditioner);
  RCP<operatort_type> belosPrec = rcp(new xpetraop_type(xpetraOperator));

  // Set up the linear equation system for Belos
  RCP<operatort_type> belosA = rcp(new xpetraop_type(systemMatrix));
  RCP<linear_problem_type> linear_problem (new linear_problem_type(belosA, solution, systemRHS));
  linear_problem->setProblem(solution, systemRHS);
  linear_problem->setRightPrec(belosPrec);

  // Build the Belos iterative solver
  solverfactory_type solverfactory;
  RCP<solver_type> solver = solverfactory.create(parameterList->get("Belos Solver Type","GMRES"), belosList);
  solver->setProblem(linear_problem);

  // Solve the linear syste
  solver->solve();

  auto print_out = Teuchos::getFancyOStream (Teuchos::rcpFromRef(std::cout));
  solution->describe(*print_out, Teuchos::VERB_EXTREME);

  return 0;
}

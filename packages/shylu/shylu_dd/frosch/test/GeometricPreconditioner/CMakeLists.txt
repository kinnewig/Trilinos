TRIBITS_ADD_EXECUTABLE(
	geometricpreconditioner
	SOURCES main.cpp
)

###########################
# GeometricPreconditioner #
###########################
## Tpetra
TRIBITS_ADD_TEST(
interfacesets
NAME test_interfacesets_DIM2_CSCommCrsMatrix_TPETRA
ARGS "--DIM=2 --M=4 --COMMSTRAT=0 --USETPETRA"
COMM mpi
NUM_MPI_PROCS 4
)

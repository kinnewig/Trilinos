TRIBITS_ADD_EXECUTABLE(
	graphtools
	SOURCES main.cpp
)

IF(HAVE_SHYLU_DDFROSCH_EPETRA AND NOT Tpetra_DefaultNode STREQUAL "Tpetra::KokkosCompat::KokkosCudaWrapperNode")
TRIBITS_ADD_TEST(
graphtools
NAME test_graphtools_EPETRA
ARGS "--USEEPETRA"
COMM mpi
NUM_MPI_PROCS 4
)
ENDIF()

TRIBITS_ADD_TEST(
graphtools
NAME test_graphtools_TPETRA
ARGS "--USETPETRA"
COMM mpi
NUM_MPI_PROCS 4
)

TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})
TRIBITS_INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/../../Utilities)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  performance_field_accessors
  SOURCES Performance_FieldAccessors.cpp
  TESTONLYLIBS phalanx_unit_test_main phalanx_test_utilities
  NUM_MPI_PROCS 1
  )

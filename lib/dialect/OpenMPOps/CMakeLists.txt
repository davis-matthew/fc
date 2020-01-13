add_library(MLIROpenMPOps
  OpenMPOps.cpp
  DialectRegistration.cpp
  )
add_dependencies(MLIROpenMPOps MLIROpenMPOpsIncGen MLIRIR MLIRStandardOps)
target_link_libraries(MLIROpenMPOps MLIRIR MLIRStandardOps)


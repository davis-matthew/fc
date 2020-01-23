add_library(MLIRFCOps
  FCOps.cpp
  DialectRegistration.cpp

  )
add_dependencies(MLIRFCOps MLIRFCOpsIncGen MLIRIR MLIRStandardOps)
target_link_libraries(MLIRFCOps MLIRIR MLIRStandardOps)


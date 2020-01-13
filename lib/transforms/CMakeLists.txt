add_library(FCMLIRTransforms
    ForOpConverter.cpp
    FCDoConverter.cpp
    MemToReg.cpp
    IteratedDominanceFrontier.cpp
    OpenMPLowering.cpp
    SimplifyCFG.cpp
    AliasAnalysis.cpp
    SimplifyLoopMemOps.cpp
    LoopStructuresToStdLowering.cpp
    ArrayOpsLowering.cpp
    LowerProgramUnit.cpp
    NestedPUVariableLowering.cpp
)

target_link_libraries(FCMLIRTransforms
 FCCommon
 MLIRAffineOps
 MLIRFCOps
 MLIRLLVMIR
 MLIRStandardOps
 MLIRAffineToStandard
 )

install(TARGETS FCMLIRTransforms DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

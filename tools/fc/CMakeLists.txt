add_executable(fc
    fc.cpp
)

set(LIBS
  MLIRAffineOps
  MLIRLLVMIR
  MLIRStandardOps
  MLIRLoopOps
)

whole_archive_link(fc ${LIBS})

target_link_libraries(fc
    FCCommon
    FCLex
    FCParse
    FCParseTree
    FCSema
    FCCodeGen
    FCMLIRTransforms
    FCToLLVM
    LLVMCore
    LLVMSupport
    LLVMBitWriter
    LLVMipo
    LLVMCodeGen
    LLVMMC
    LLVMX86Desc
    LLVMTarget
    LLVMX86Info
    LLVMAnalysis
    LLVMX86CodeGen
    LLVMX86AsmParser
    MLIRTargetLLVMIR
    MLIRStandardToLLVM
    ${LIBS}
)

install(TARGETS fc DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

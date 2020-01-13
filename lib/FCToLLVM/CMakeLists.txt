add_library(FCToLLVM
    FCRuntimeHelper.cpp
    FCToLLVMLowering.cpp
    FCFuncOpLowering.cpp
    FCTypeConverter.cpp
)

target_link_libraries(FCToLLVM
 FCCommon
 MLIRAffineOps
 MLIRFCOps
 MLIRLLVMIR 
 MLIRStandardOps
 MLIRAffineToStandard
 )

install(TARGETS FCToLLVM DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

add_library(FCCodeGen
    CodeGen.cpp
    CGStmt.cpp
    CGExpr.cpp
    CGOpenMP.cpp
    CGASTHelper.cpp
)

target_link_libraries(FCCodeGen
 FCCommon
 FCParseTree
 MLIRAffineOps
 MLIRFCOps
 MLIROpenMPOps
 MLIRLLVMIR
 MLIRStandardOps
 MLIRAffineToStandard
 )

install(TARGETS FCCodeGen DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

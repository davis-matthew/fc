add_library(FCParse 
    Parser.cpp
    ParseCommon.cpp
    ParseDecl.cpp
    ParseExpr.cpp
    ParseOpenMP.cpp
    ParseStmt.cpp
    ParserUtils.cpp
)

target_link_libraries(FCParse
 FCLex
 FCCommon)

install(TARGETS FCParse DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

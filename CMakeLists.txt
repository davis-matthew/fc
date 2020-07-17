cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(fc)

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Debug )
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}" )

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using LLVM install directory ${LLVM_BINARY_DIR}")
message(STATUS "Using LLVM build directory ${LLVM_BUILD_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
include(AddLLVM)
include(LLVMConfig)
 

# TODO: Use llvm-config to get the data.
if(NOT LLVM_BUILD_DIR)
  message(FATAL_ERROR "LLVM_BUILD_DIR not found")
endif()
set (LLVM_BUILD_TOOLS_DIR ${LLVM_BUILD_DIR}/bin)
set(LLVM_EXTERNAL_LIT ${LLVM_BUILD_TOOLS_DIR}/llvm-lit CACHE STRING "Command used to spawn lit")
message(STATUS "LIT ${LLVM_EXTERNAL_LIT}")

#TODO: This is to handle the current static registration, but should be
# factored out a bit.
function(whole_archive_link target)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(link_flags "-L${LLVM_BUILD_DIR}/lib ")
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-Wl,-force_load ${CMAKE_BINARY_DIR}/lib/lib${LIB}.a ")
    ENDFOREACH(LIB)
  elseif(MSVC)
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "/WHOLEARCHIVE:${LIB} ")
    ENDFOREACH(LIB)
  else()
    set(link_flags "-L${LLVM_BINARY_DIR}/lib -Wl,--whole-archive,")
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-l${LIB},")
    ENDFOREACH(LIB)
    string(CONCAT link_flags ${link_flags} "--no-whole-archive")
  endif()
  set_target_properties(${target} PROPERTIES LINK_FLAGS ${link_flags})
endfunction(whole_archive_link)

add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Ignore unhandled switch cases for now.
# Can be moved to C++11 . There are one or places which requires c++14. This can be easily changed.
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -fno-rtti -std=c++14 -Wno-null-conversion -Wno-unused-command-line-argument" )

include_directories(${CMAKE_CURRENT_SOURCE_DIR} include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

#TODO Take mlir-tblgen path from cmake
set(MLIR_TABLEGEN_EXE ${LLVM_BINARY_DIR}/bin/mlir-tblgen)

include(TableGen)

function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV} "-I${MLIR_MAIN_SRC_DIR}" "-I${LLVM_BINARY_DIR}/include" "-I${PROJECT_SOURCE_DIR}/include")
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

# Generate a CompilationDatabase (compile_commands.json file) for our build,
# for use by clang_complete, YouCompleteMe, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(FC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FC_TOOLS_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_subdirectory(include/dialect)
add_subdirectory(lib)
add_subdirectory(runtime)
add_subdirectory(tools)
add_subdirectory(test)
add_subdirectory(utils)

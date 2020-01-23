set(LLVM_TARGET_DEFINITIONS FCOps.td)
mlir_tablegen(FCOps.h.inc -gen-op-decls)
mlir_tablegen(FCOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRFCOpsIncGen)

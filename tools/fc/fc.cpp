// Copyright (c) 2019, Compiler Tree Technologies Pvt Ltd.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include "AST/ASTContext.h"
#include "AST/ProgramUnit.h"
#include "codegen/CGASTHelper.h"
#include "codegen/CodeGen.h"
#include "common/Debug.h"
#include "common/Diagnostics.h"
#include "common/Source.h"
#include "lex/Lexer.h"
#include "parse/Parser.h"
#include "sema/Sema.h"
#include "transforms/FCMLIRPass.h"

#include "dialect/FC/FCOps.h"
#include "dialect/OpenMP/OpenMPOps.h"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include <fstream>
#include <streambuf>
#include <string>

using namespace llvm;

using namespace fc;

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::init("-"));

enum Action {
  EmitRawAST,
  EmitAST,
  EmitIR,
  EmitMLIR,
  EmitLoopOpt,
  EmitLLVM,
  EmitBC,
  EmitASM,
  EmitExe,
};

cl::opt<Action> ActionOpt(
    cl::desc("Choose IR Type:"),
    cl::values(clEnumValN(EmitAST, "emit-ast",
                          "Emit AST (after semantic checks)"),
               clEnumValN(EmitRawAST, "emit-raw-ast",
                          "Emit AST (before semantic checks)"),
               clEnumValN(EmitMLIR, "emit-mlir", "Emit MLIR"),
               clEnumValN(EmitIR, "emit-ir", "Emit FCIR"),
               clEnumValN(EmitLoopOpt, "loop-opt", "Emit MLIR after Loop Opts"),
               clEnumValN(EmitLLVM, "emit-llvm", "Emit LLVM IR"),
               clEnumValN(EmitBC, "emit-bc", "Emit LLVM BC"),
               clEnumValN(EmitASM, "emit-asm", "Emit ASM")),
    cl::init(EmitExe));

cl::opt<Standard>
    FortranStandard("std", cl::desc("Choose fortran standard:"),
                    cl::values(clEnumVal(f77, "Fortran77 standard"),
                               clEnumVal(f95, "Fortran95 standard (default)")),
                    cl::init(Standard::None));

enum OptLevel {
  O0 = 0,
  O1,
  O2,
  O3,
};

cl::opt<OptLevel> OptimizationLevel(
    cl::desc("Choose optimization level:"),
    cl::values(clEnumVal(O0, "No optimization"),
               clEnumVal(O1, "Enable trivial optimizations"),
               clEnumVal(O2, "Enable default optimizations"),
               clEnumVal(O3, "Enable expensive optimizations")),
    cl::init(O1));

cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"),
                                    cl::value_desc("filename"), cl::init(""));

cl::opt<std::string> RuntimePath("L", cl::desc("Specify FC runtime path"),
                                 cl::value_desc("<path-to-fc-runtime>"),
                                 cl::init(""));

llvm::cl::opt<bool> StopAtCompile("c", llvm::cl::desc("stop at compilation"),
                                  llvm::cl::init(false));

llvm::cl::opt<bool> EnableDebug("g", llvm::cl::desc("Enable debugging symbols"),
                                llvm::cl::init(false));

llvm::cl::opt<bool> PrintMLIRPasses("print-mlir",
                                    llvm::cl::desc("Print after all"),
                                    llvm::cl::init(false));

llvm::cl::opt<bool> EnableLNO("lno",
                              llvm::cl::desc("Enable Loop Nest Optimizations"),
                              llvm::cl::init(false));

llvm::cl::opt<bool> DumpVersion("v", llvm::cl::desc("Version check"),
                                llvm::cl::init(false));

llvm::cl::opt<bool> PrepareForLTO("flto", llvm::cl::desc("Prepare for LTO"),
                                  llvm::cl::init(false));

llvm::cl::opt<std::string> MArchName("march",
                                     cl::desc("Specify target architecture"),
                                     cl::value_desc("marchname"), cl::init(""));

static bool prepareLLVMTarget(std::unique_ptr<llvm::TargetMachine> &TM,
                              std::unique_ptr<llvm::Module> &llvmModule) {
  // Initialize targets.
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmParser();
  LLVMInitializeX86AsmPrinter();

  llvm::StringRef ModuleName{InputFilename};
  llvmModule->setSourceFileName(InputFilename);

  // set LLVM target triple.
  // Default to x86_64 for now.
  auto TargetTriple = sys::getDefaultTargetTriple();
  llvmModule->setTargetTriple(TargetTriple);

  std::string Error;
  std::string Triple = llvmModule->getTargetTriple();
  const llvm::Target *TheTarget = TargetRegistry::lookupTarget(Triple, Error);
  if (!TheTarget) {
    error() << "\n could not find target for triple " << Triple;
    return false;
  }

  llvm::Optional<CodeModel::Model> CM = llvm::CodeModel::Small;
  llvm::Optional<Reloc::Model> RM = llvm::Reloc::Static;
  auto OptLevel = CodeGenOpt::Default;
  switch (OptimizationLevel) {
  case OptLevel::O1:
    OptLevel = CodeGenOpt::Less;
    break;
  case OptLevel::O2:
  case OptLevel::O0:
    OptLevel = CodeGenOpt::Default;
    break;
  case OptLevel::O3:
    OptLevel = CodeGenOpt::Aggressive;
    break;
  };

  llvm::TargetOptions Options;
  Options.ThreadModel = llvm::ThreadModel::POSIX;
  Options.FloatABIType = llvm::FloatABI::Default;
  Options.AllowFPOpFusion = llvm::FPOpFusion::Standard;
  Options.UnsafeFPMath = true;

  const char *CPU;
  if (MArchName.empty())
    CPU = sys::getHostCPUName().data();
  else
    CPU = MArchName.c_str();
  StringMap<bool> HostFeatures;
  auto status = sys::getHostCPUFeatures(HostFeatures);
  SubtargetFeatures features;
  if (status) {
    for (auto &F : HostFeatures) {
      features.AddFeature(F.first(), F.second);
    }
  }
  TM.reset(TheTarget->createTargetMachine(Triple, CPU, features.getString(),
                                          Options, RM, CM, OptLevel));

  llvmModule->setDataLayout(TM->createDataLayout());
  return true;
}

// FIXME: Fix LLVM Module for linkage and alias metadata info. These are
// things should be propagated from MLIR to LLVM dialect.
static void fixLLVMModule(llvm::Module *llvmModule) {
  llvm::MDBuilder mbuilder(llvmModule->getContext());

  std::function<void(llvm::Value *, llvm::MDNode *, llvm::MDNode *,
                     llvm::MDNode *)>
      addMetadata = [&](llvm::Value *val, llvm::MDNode *readScope,
                        llvm::MDNode *writeScope, llvm::MDNode *noalias) {
        if (isa<llvm::ExtractValueInst>(val) ||
            isa<llvm::GetElementPtrInst>(val)) {
          if (val->getType()->isPointerTy()) {
            for (auto user : val->users()) {
              addMetadata(user, readScope, writeScope, noalias);
            }
          }
          return;
        }
        if (auto load = llvm::dyn_cast<llvm::LoadInst>(val)) {
          load->setMetadata(LLVMContext::MD_alias_scope, readScope);
          load->setMetadata(LLVMContext::MD_noalias, noalias);
        } else if (auto store = llvm::dyn_cast<llvm::StoreInst>(val)) {
          store->setMetadata(LLVMContext::MD_alias_scope, writeScope);
          store->setMetadata(LLVMContext::MD_noalias, noalias);
        }
      };

  for (auto &F : *llvmModule) {
    llvm::SmallVector<llvm::MDNode *, 2> readScopes;
    llvm::SmallVector<llvm::MDNode *, 2> writeScopes;
    auto domain = mbuilder.createAnonymousAliasScopeDomain();
    for (auto &arg : F.args()) {
      if (arg.getType()->isPointerTy()) {
        arg.addAttr(llvm::Attribute::NoAlias);
      }
      auto read = mbuilder.createAnonymousAliasScope(domain);
      auto write = mbuilder.createAnonymousAliasScope(domain);
      readScopes.push_back(read);
      writeScopes.push_back(write);
    }
    for (auto &arg : F.args()) {
      auto argNum = arg.getArgNo();
      llvm::SmallVector<llvm::LoadInst *, 2> loads;
      llvm::MDNode *curr = nullptr;
      for (unsigned I = 0; I < readScopes.size(); ++I) {
        if (I == argNum)
          continue;
        if (!curr)
          curr = readScopes[I];
        else
          curr = llvm::MDNode::concatenate(readScopes[I], curr);
        curr = llvm::MDNode::concatenate(writeScopes[I], curr);
      }
      for (auto user : arg.users()) {
        addMetadata(user, readScopes[argNum], writeScopes[argNum], curr);
      }
    }
  }
}

static bool runLLVMPasses(std::unique_ptr<llvm::Module> &llvmModule,
                          std::unique_ptr<llvm::TargetMachine> &TM,
                          llvm::raw_fd_ostream &OS) {
  prepareLLVMTarget(TM, llvmModule);
  fixLLVMModule(llvmModule.get());

  if (ActionOpt == EmitBC && OptimizationLevel == O0) {
    FC_DEBUG(debug() << "Emitting LLVM BC before optimizations\n");
    llvm::WriteBitcodeToFile(*llvmModule.get(), OS);
    OS.flush();
    return true;
  }

  if (ActionOpt == EmitLLVM && OptimizationLevel == O0) {
    FC_DEBUG(debug() << "Emitting LLVM IR\n");
    llvmModule->print(OS, nullptr);
    OS.flush();
    return true;
  }

  llvm::Triple TargetTriple(llvmModule->getTargetTriple());
  std::unique_ptr<TargetLibraryInfoImpl> TLII(
      new TargetLibraryInfoImpl(TargetTriple));

  PassManagerBuilder PMBuilder;
  PMBuilder.OptLevel = OptimizationLevel;
  PMBuilder.SizeLevel = 0;
  PMBuilder.LoopVectorize = OptimizationLevel > 1;
  PMBuilder.SLPVectorize = OptimizationLevel > 1;
  PMBuilder.PrepareForLTO = PrepareForLTO;
  PMBuilder.DisableUnrollLoops = !(OptimizationLevel > 1);
  PMBuilder.Inliner = createFunctionInliningPass(PMBuilder.OptLevel,
                                                 PMBuilder.SizeLevel, false);

  legacy::FunctionPassManager FPM(llvmModule.get());
  FPM.add(new TargetLibraryInfoWrapperPass(*TLII));
  FPM.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  legacy::PassManager MPM;
  MPM.add(new TargetLibraryInfoWrapperPass(*TLII));
  MPM.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  PMBuilder.populateFunctionPassManager(FPM);
  PMBuilder.populateModulePassManager(MPM);

  // Run all the function passes.
  FPM.doInitialization();
  for (llvm::Function &F : *llvmModule)
    if (!F.isDeclaration())
      FPM.run(F);
  FPM.doFinalization();

  // Run all the module passes.
  MPM.run(*llvmModule.get());

  if (PrepareForLTO || ActionOpt == EmitBC) {
    FC_DEBUG(debug() << "Emitting LLVM BC after optimizations\n");
    llvm::WriteBitcodeToFile(*llvmModule, OS);
    OS.flush();
    return true;
  }

  if (ActionOpt == EmitLLVM) {
    FC_DEBUG(debug() << "Emitting LLVM IR after optimizations\n");
    llvmModule->print(OS, nullptr);
    OS.flush();
    return true;
  }
  return true;
}

// High level MLIR passes.
static void addFCDialectPasses(mlir::PassManager &mlirPM) {
  mlirPM.addPass(createMemToRegPass());
  mlirPM.addPass(createSimplifyLoopMemOperations());
}

// High level MLIR lowering.
static void addLowerFCDialectPasses(mlir::PassManager &mlirPM) {
  mlirPM.addPass(createNestedPUVariableLoweringPass());
  mlirPM.addPass(createProgramUnitLoweringPass());
}

static void addLowerLevelMLIROptPasses(mlir::PassManager &mlirPM) {
  mlirPM.addPass(mlir::createCanonicalizerPass());
  mlirPM.addPass(createCSEPass());
  mlirPM.addPass(mlir::createLoopInvariantCodeMotionPass());
  mlirPM.addPass(mlir::createCanonicalizerPass());
  mlirPM.addPass(createSimplifyCFGPass());
  if (EnableLNO) {
    mlirPM.addPass(createForOpConverterPass());
    mlirPM.addPass(createSimplifyLoopMemOperations());
    mlirPM.addPass(createFCDoConverterPass());
    //mlirPM.addPass(createLNODriverPass());
  }
  mlirPM.addPass(mlir::createLoopFusionPass(2, 1000, true));
  mlirPM.addPass(mlir::createLoopUnrollAndJamPass(-1));
  mlirPM.addPass(mlir::createCanonicalizerPass());
  mlirPM.addPass(mlir::createCSEPass());
}

static void addLLVMLoweringPasses(mlir::PassManager &mlirPM) {
  mlirPM.addPass(createArrayOpsLoweringPass());
  mlirPM.addPass(createSimplifyCFGPass());
  mlirPM.addPass(createLoopStructureLoweringPass());
  mlirPM.addPass(createSimplifyCFGPass());
  mlirPM.addPass(createFCToLLVMLoweringPass());
  mlirPM.addPass(createOpenMPLoweringPass());
}

static bool runMLIRPasses(mlir::OwningModuleRef &theModule,
                          llvm::raw_fd_ostream &OS) {

  mlir::PassManager mlirPM(theModule->getContext());
  mlirPM.disableMultithreading();
  mlir::applyPassManagerCLOptions(mlirPM);

  // Enable print after all.
  if (PrintMLIRPasses) {
    // mlir::IRPrinterConfig config;
    // mlirPM.enableIRPrinting([](mlir::Pass *p) { return false; },
    //                         [](mlir::Pass *p) { return true; }, true,
    //                         llvm::errs());
  }

  switch (ActionOpt) {
  case EmitIR: {
    if (OptimizationLevel > 0) {
      addFCDialectPasses(mlirPM);
    }
    break;
  }
  case EmitMLIR:
  case EmitLoopOpt: {
    if (OptimizationLevel > 0 || EnableLNO) {
      addFCDialectPasses(mlirPM);
    }
    addLowerFCDialectPasses(mlirPM);
    if (OptimizationLevel > 0 || EnableLNO) {
      addLowerLevelMLIROptPasses(mlirPM);
    }
    break;
  }
  default: {
    if (OptimizationLevel > 0 || EnableLNO) {
      addFCDialectPasses(mlirPM);
    }
    addLowerFCDialectPasses(mlirPM);
    if (OptimizationLevel > 0 || EnableLNO) {
      addLowerLevelMLIROptPasses(mlirPM);
    }
    addLLVMLoweringPasses(mlirPM);
  }
  };

  auto result = mlirPM.run(theModule.get());
  if (failed(result)) {
    llvm::errs() << "Failed to run MLIR Pass manager\n";
    return false;
  }
  return true;
}

static void fixFlags(StringRef InputFile) {
  if (FortranStandard == Standard::None) {
    auto extension = llvm::sys::path::extension(InputFile);
    if (extension == ".f")
      FortranStandard = f77;
    else
      FortranStandard = f95;
  }

  if (PrepareForLTO) {
    ActionOpt = EmitBC;
  }
  if (ActionOpt == EmitLoopOpt) {
    EnableLNO = true;
  }

  if (OutputFilename == "") {
    std::string extension = "";
    switch (ActionOpt) {
    case EmitRawAST:
    case EmitAST:
      extension = "ast.c";
      break;
    case EmitBC:
      if (PrepareForLTO) {
        extension = "o";
      } else {
        extension = "bc";
      }
      break;
    case EmitMLIR:
    case EmitIR:
    case EmitLoopOpt:
      extension = "mlir";
      break;
    case EmitLLVM:
      extension = "ll";
      break;
    case EmitASM:
      extension = "s";
      break;
    case EmitExe:
      if (StopAtCompile) {
        extension = "o";
      } else {
        OutputFilename = "a.out";
      }
      break;
    default:
      llvm_unreachable("Unhandled action type");
    };
    if (ActionOpt != EmitExe || (ActionOpt == EmitExe && StopAtCompile)) {
      // Replace the existing extension in the input file to the new one.
      assert(!extension.empty());
      OutputFilename = InputFile;
      llvm::SmallString<128> outputFile(OutputFilename);
      llvm::sys::path::replace_extension(outputFile, extension);
      OutputFilename = outputFile.str();
    }
  }
}

static bool compileFile(StringRef InputFile) {
  fixFlags(InputFile);

  FC_DEBUG(debug() << "Started parsing input file " << InputFile << "\n");

  auto file = llvm::MemoryBuffer::getFileOrSTDIN(InputFile);
  if (!file) {
    error() << "Failed to read input file\n";
    return false;
  }
  auto fileRef = file.get()->getBuffer();

  SourceLoc loc;
  Lexer lexer(InputFilename, FortranStandard, loc, fileRef.begin(),
              fileRef.end());
  Diagnostics diagEngine(InputFilename);
  ASTContext astContext(diagEngine, InputFile);
  parser::Parser parser(astContext, lexer, diagEngine);

  // Run parser and generate AST.
  FC_DEBUG(debug() << "Started running the Parser\n");
  bool status = parser.parseProgram();
  FC_DEBUG(debug() << "Done with the Parser\n");
  if (!status)
    return false;

  // Runs the semantic check on parser.
  Sema sema(parser.getTree());
  std::error_code EC;
  llvm::raw_fd_ostream OS(OutputFilename, EC, llvm::sys::fs::F_None);
  if (ActionOpt == EmitRawAST) {
    FC_DEBUG(debug() << "Emitting the raw AST\n");
    parser.dumpParseTree(OS);
    OS.flush();
    return true;
  }

  status = sema.run();
  if (!status) {
    error() << "\n Error while running Sema";
    return false;
  }

  if (ActionOpt == EmitAST) {
    FC_DEBUG(debug() << "Emitting the AST after SEMA\n");
    parser.dumpParseTree(OS);
    OS.flush();
    return true;
  }

  mlir::registerDialect<FC::FCOpsDialect>();
  mlir::registerDialect<OMP::OpenMPDialect>();
  mlir::MLIRContext mlirContext;
  mlir::OwningModuleRef theModule;
  mlir::registerPassManagerCLOptions();
  ASTPassManager astPassManager(parser.getTree(), "AST CodeGen PassManager");
  auto mlirCG =
      new CodeGen(astContext, theModule, mlirContext, FortranStandard);
  astPassManager.addPass(mlirCG);

  // MLIR Codegen.
  if (!astPassManager.run()) {
    error() << "\n Error during MLIR emission";
    return false;
  }

  // Run transforms on MLIR.
  if (!runMLIRPasses(theModule, OS)) {
    llvm::errs() << "Failed to emit MLIR\n";
    return false;
  }

  bool emitMLIRCode = false;
  switch (ActionOpt) {
  case EmitIR:
  case EmitMLIR:
  case EmitLoopOpt:
    emitMLIRCode = true;
    break;
  default:
    emitMLIRCode = false;
  };

  if (emitMLIRCode) {
    theModule->print(OS);
    OS.flush();
    return true;
  }

  // Emit LLVM IR from MLIR.
  auto llvmModule = mlir::translateModuleToLLVMIR(theModule.get());
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return false;
  }

  // Prepare and run LLVM IR passes.
  std::unique_ptr<llvm::TargetMachine> TM;
  if (!runLLVMPasses(llvmModule, TM, OS)) {
    llvm::errs() << "\n Failed to run LLVM Passes";
    return false;
  }

  // runLLVMPasses already dumped ir
  if (ActionOpt == EmitBC || ActionOpt == EmitLLVM || PrepareForLTO)
    return true;

  // Create CodeGen Passes.
  legacy::PassManager CGPasses;
  CGPasses.add(createTargetTransformInfoWrapperPass(TM->getTargetIRAnalysis()));

  if (ActionOpt == EmitASM) {
    if (TM->addPassesToEmitFile(CGPasses, OS, nullptr,
                                llvm::CGFT_AssemblyFile)) {
      error() << "\n Failed to emit Assembly file.";
      return false;
    }

    // Run all codegen passes.
    CGPasses.run(*llvmModule.get());
    FC_DEBUG(debug() << "Emitting ASM file\n");
    return true;
  }

  // Generate binary action. Emit object file first and then create exe.
  assert(ActionOpt == EmitExe);
  LLVMTargetMachine &LTM = static_cast<LLVMTargetMachine &>(*TM);
  llvm::MachineModuleInfo MMI(&LTM);
  auto MCContext = &MMI.getContext();

  std::string objFile = "";

  // Create temporary object file.
  InputFile = llvm::sys::path::filename(InputFilename);
  std::string tempFilename = "/tmp/" + InputFile.str();
  auto TmpFile = llvm::sys::fs::TempFile::create(tempFilename + "-%%%%%.o");
  if (!TmpFile) {
    error() << "\n Failed to create temporary file!";
    return false;
  }

  objFile = StopAtCompile ? OutputFilename : TmpFile.get().TmpName;
  llvm::raw_fd_ostream TOS(objFile, EC, llvm::sys::fs::F_None);

  FC_DEBUG(debug() << "Emitting Temp file " << objFile);
  if (TM->addPassesToEmitMC(CGPasses, MCContext, TOS, false)) {
    error() << "\n Failed to generate object code";
    return false;
  }

  // Run all codegen passes.
  CGPasses.run(*llvmModule.get());

  if (StopAtCompile)
    return true;
  // Create ld command

  // FIXME: Expects clang binary for linking.
  StringRef ldCommand = getenv("CLANG_BINARY");
  if (ldCommand.empty()) {
    error() << "\n CLANG_BINARY env variable not set!";
    return false;
  }

  if (RuntimePath.size() > 0)
    RuntimePath = "-L" + RuntimePath;
  const char *args[9] = {ldCommand.data(),
                         objFile.c_str(),
                         "-o",
                         OutputFilename.c_str(),
                         "-lFC",
                         "-lm",
                         "-lomp",
                         RuntimePath.c_str(),
                         NULL};

  std::string errorStr;
  bool ExecFailed = false;
  std::vector<llvm::Optional<StringRef>> Redirects;
  Redirects = {llvm::NoneType::None, llvm::NoneType::None,
               llvm::NoneType::None};

  llvm::SmallVector<llvm::StringRef, 8> argsArr(args, args + 8);
  llvm::sys::ExecuteAndWait(ldCommand, argsArr, llvm::None, Redirects, 0, 0,
                            &errorStr, &ExecFailed);
  if (ExecFailed) {
    error() << "\n ld tool execution failed : " << errorStr;
    return false;
  }
  FC_DEBUG(debug() << "Emitting binary file" << OutputFilename);
  // Delete the temp file created.
  if (auto E = TmpFile->discard()) {
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  if (DumpVersion) {
    std::cout << "\nFortran Compiler by Compiler Tree Technologies Ltd";
    std::cout << "\nVersion : 0.2\n";
    return 0;
  }
  return !compileFile(InputFilename);
}

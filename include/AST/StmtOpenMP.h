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
#ifndef FC_STMTOPENMP_H
#define FC_STMTOPENMP_H

#include "AST/ParserTreeCommon.h"
#include "AST/Type.h"
#include "common/Source.h"

#include "AST/Expressions.h"
#include "AST/ProgramUnit.h"
#include "AST/Stmt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

namespace fc {
namespace ast {

// TODO : Add a base class for all openmp stmts
class OpenMPParallelStmt : public Stmt {
private:
  SymbolSet mapedSymbols;

protected:
  friend class ParseTreeBuilder;
  OpenMPParallelStmt(SourceLoc _sourceLoc, Block *block)
      : Stmt(OpenMPParallelStmtKind, _sourceLoc) {
    setOperands({block});
    setParentForOperands();
  }

public:
  constexpr static bool classof(const Stmt *Stmt) {
    // Add here when you add more OpenMPKinds
    return Stmt->getStmtType() == OpenMPParallelStmtKind;
  }

  std::string dump(llvm::raw_ostream &OS, int level = 0) const;

  inline Block *getBlock() const { return static_cast<Block *>(operands[0]); }

  ~OpenMPParallelStmt() {}

  void getMapedSymbols(SymbolSet &list) {
    list.clear();
    list.insert(mapedSymbols.begin(), mapedSymbols.end());
  }

  void insertMapedSymbol(Symbol *symbol) { mapedSymbols.insert(symbol); }
};

class OpenMPParallelDoStmt : public Stmt {
private:
  SymbolSet mapedSymbols;

protected:
  friend class ParseTreeBuilder;
  OpenMPParallelDoStmt(SourceLoc _sourceLoc, DoStmt *dostmt)
      : Stmt(OpenMPParallelDoStmtKind, _sourceLoc) {
    setOperands({dostmt});
    setParentForOperands();
  }

public:
  constexpr static bool classof(const Stmt *Stmt) {
    // Add here when you add more OpenMPKinds
    return Stmt->getStmtType() == OpenMPParallelDoStmtKind;
  }

  std::string dump(llvm::raw_ostream &OS, int level = 0) const;

  inline DoStmt *getDoStmt() const {
    return static_cast<DoStmt *>(operands[0]);
  }

  ~OpenMPParallelDoStmt() {}

  void getMapedSymbols(SymbolSet &list) {
    list.clear();
    list.insert(mapedSymbols.begin(), mapedSymbols.end());
  }

  void insertMapedSymbol(Symbol *symbol) { mapedSymbols.insert(symbol); }
};

class OpenMPSingleStmt : public Stmt {

protected:
  friend class ParseTreeBuilder;
  OpenMPSingleStmt(SourceLoc _sourceLoc, Block *block)
      : Stmt(OpenMPSingleStmtKind, _sourceLoc) {
    setOperands({block});
    setParentForOperands();
  }

public:
  constexpr static bool classof(const Stmt *Stmt) {
    return Stmt->getStmtType() == OpenMPSingleStmtKind;
  }

  std::string dump(llvm::raw_ostream &OS, int level = 0) const;

  inline Block *getBlock() const { return static_cast<Block *>(operands[0]); }

  ~OpenMPSingleStmt() {}
};

class OpenMPMasterStmt : public Stmt {
private:
protected:
  friend class ParseTreeBuilder;
  OpenMPMasterStmt(SourceLoc _sourceLoc, Block *block)
      : Stmt(OpenMPMasterStmtKind, _sourceLoc) {
    setOperands({block});
    setParentForOperands();
  }

public:
  constexpr static bool classof(const Stmt *Stmt) {
    return Stmt->getStmtType() == OpenMPMasterStmtKind;
  }

  std::string dump(llvm::raw_ostream &OS, int level = 0) const;

  inline Block *getBlock() const { return static_cast<Block *>(operands[0]); }

  ~OpenMPMasterStmt() {}
};

class OpenMPDoStmt : public Stmt {
private:
  SymbolList mapedSymbols;

protected:
  friend class ParseTreeBuilder;
  OpenMPDoStmt(SourceLoc _sourceLoc, DoStmt *dostmt)
      : Stmt(OpenMPDoStmtKind, _sourceLoc) {
    setOperands({dostmt});
    setParentForOperands();
  }

public:
  constexpr static bool classof(const Stmt *Stmt) {
    return Stmt->getStmtType() == OpenMPDoStmtKind;
  }

  std::string dump(llvm::raw_ostream &OS, int level = 0) const;

  inline DoStmt *getDoStmt() const {
    return static_cast<DoStmt *>(operands[0]);
  }

  ~OpenMPDoStmt() {}
};
} // namespace ast
} // namespace fc
#endif

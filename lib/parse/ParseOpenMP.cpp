// Copyright (c) 2019, Compiler Tree Technologies Pvt Ltd.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include "AST/Statements.h"
#include "AST/StmtOpenMP.h"
#include "parse/Parser.h"
#include "sema/Intrinsics.h"

using namespace fc;
using namespace parser;

Stmt *Parser::parseOpenMPRegion() {
  if (match({tok::kw_omp, tok::kw_parallel, tok::kw_do}))
    return parseOpenMPParallelDoStmt();
  if (match({tok::kw_omp, tok::kw_parallel}))
    return parseOpenMPParallel();
  llvm_unreachable("unhandled openmp region");
}

OpenMPParallelStmt *Parser::parseOpenMPParallel() {
  SourceLoc loc = getCurrLoc();
  assert(is(tok::dir));

  if (!expect({tok::kw_omp, tok::kw_parallel})) {
    Diag.printError(getCurrLoc(), diag::err_ompstmt);
    return nullptr;
  }
  consumeToken(tok::kw_parallel);
  assert(is(tok::eol));
  consumeToken(tok::eol);
  auto ompBody = parseBlock();
  assert(ompBody);
  assert(is(tok::dir));
  if (!expect({tok::kw_omp, tok::kw_end, tok::kw_parallel, tok::eol})) {
    Diag.printError(getCurrLoc(), diag::err_ompstmt);
    return nullptr;
  }
  assert(is(tok::eol));
  consumeToken(tok::eol);
  return builder.buildOpenMPParallelStmt(ompBody, loc);
}

OpenMPParallelDoStmt *Parser::parseOpenMPParallelDoStmt() {
  SourceLoc loc = getCurrLoc();
  assert(is(tok::dir));
  llvm::StringRef name = "";

  if (!expect({tok::kw_omp, tok::kw_parallel, tok::kw_do, tok::eol})) {
    Diag.printError(getCurrLoc(), diag::err_ompstmt);
    return nullptr;
  }
  assert(is(tok::eol));
  consumeToken(tok::eol);
  assert(is(tok::kw_do));
  consumeToken(tok::kw_do);
  auto dostmt = parseDoStmt(name, loc);
  assert(dostmt);
  consumeToken(tok::eol);
  assert(is(tok::dir));
  if (!expect(
          {tok::kw_omp, tok::kw_end, tok::kw_parallel, tok::kw_do, tok::eol})) {
    Diag.printError(getCurrLoc(), diag::err_ompstmt);
    return nullptr;
  }
  assert(is(tok::eol));
  consumeToken(tok::eol);
  return builder.buildOpenMPParallelDoStmt(dostmt, loc);
}

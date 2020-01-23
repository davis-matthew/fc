
def FC_ComplexConstantOp : FC_Op<"complex_constant", [NoSideEffect]> {
  let summary = "complex constant";
  let description = [{
    Constant operation turns a literal complex type value into an SSA value.
  }];

  let arguments = (ins ArrayAttr:$value);
  let results = (outs AnyComplex);
  let verifier = ?;
  let printer = [{
    p << "fc.complex_constant " << value();
    p << " : " << getResult().getType();
  }];
  let parser = ? ;
}

class FC_ComplexBinOp<string mnemonic, list<OpTrait> traits = []> :
  FC_Op<mnemonic, traits> {

  let summary = "Element wise binary operation on complex type";
  let description = [{

  }];

  let arguments = (ins AnyComplex:$lhs, AnyComplex:$rhs);
  let results = (outs AnyComplex);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value lhs, Value rhs",
    [{
      result.addOperands({lhs, rhs});
      result.types.push_back(rhs.getType());
    }]>];

  // TODO: add compatibility tests.
  let verifier = ? ;
  let printer = [{
    p << getOperationName() << " " << lhs() << ", " << rhs();
    p.printOptionalAttrDict(getAttrs());
    p << " : " << getResult().getType();
  }];
  let parser = ? ;
}

def FC_ComplexAddOp : FC_ComplexBinOp<"complex_add", [Commutative]>;
def FC_ComplexSubOp : FC_ComplexBinOp<"complex_sub", []>;
def FC_ComplexMulOp : FC_ComplexBinOp<"complex_mul", [Commutative]>;
def FC_ComplexDivOp : FC_ComplexBinOp<"complex_div", []>;

def FC_ComplexConjugateOp : FC_Op<"complex_conjugate"> {
  let summary = "returns conjugate of complex number";
  let description = [{

  }];

  let arguments = (ins AnyComplex:$inval);
  let results = (outs AnyComplex);

  let verifier = ? ;
  let printer = [{
    p << "fc.complex_conjugate " << inval();
    p.printOptionalAttrDict(getAttrs());
    p << " : " << getResult().getType();
  }];
  let parser = ? ;
}

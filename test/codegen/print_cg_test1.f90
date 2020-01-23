! RUN: %fc -emit-ir %s -o - | FileCheck %s
program test
print *, 10 !CHECK: fc.print
end

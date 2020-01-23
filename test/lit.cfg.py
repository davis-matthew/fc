# -*- Python -*-

import os
import platform
import re
import subprocess
import sys

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'FC'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)


# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.f90', '.f']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.fc_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))

config.environment['CLANG_BINARY'] = os.getenv('CLANG_BINARY')

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['input', 'CMakeLists.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.fc_obj_root, 'test')

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.fc_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

# For each occurrence of a fc tool name, replace it with the full path to
# the build directory holding that tool.  We explicitly specify the directories
# to search to ensure that we get the tools just built and not some random
# tools that might happen to be in the user's PATH.
tool_dirs = [config.llvm_tools_dir, config.fc_tools_dir]

tools = [ToolSubst('%fc', command=FindTool('fc'), unresolved='fatal')]

llvm_config.add_tool_substitutions(tools, tool_dirs)


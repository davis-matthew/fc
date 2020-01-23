macro(copy_files srcDir destDir)
    make_directory(${destDir})

    file(GLOB templateFiles RELATIVE ${srcDir} ${srcDir}/*)
    foreach(templateFile ${templateFiles})
        set(srcTemplatePath ${srcDir}/${templateFile})
        if(NOT IS_DIRECTORY ${srcTemplatePath})
            configure_file(
                    ${srcTemplatePath}
                    ${destDir}/${templateFile}
                    @ONLY)
        endif(NOT IS_DIRECTORY ${srcTemplatePath})
    endforeach(templateFile)
endmacro(copy_files)

# Copy some of the input files given for the tests.
copy_files(${CMAKE_CURRENT_SOURCE_DIR}/output/input ${CMAKE_CURRENT_BINARY_DIR}/output/input)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(FC_TEST_DEPENDS
  fc
  FC
)

# Show progress bar!
list(APPEND LLVM_LIT_ARGS
    "-s"
)

add_lit_testsuite(check-all "Running the FC regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${FC_TEST_DEPENDS}
  )

set_target_properties(check-all PROPERTIES FOLDER "Tests")

add_lit_testsuites(FC ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${FC_TEST_DEPENDS}
)

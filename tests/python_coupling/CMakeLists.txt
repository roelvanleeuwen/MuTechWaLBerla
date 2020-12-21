###################################################################################################
#
# Tests for python coupling
#
###################################################################################################

waLBerla_link_files_to_builddir( *.py )

if (WALBERLA_BUILD_WITH_PYTHON)
    waLBerla_compile_test( FILES ConfigFromPythonTest.cpp DEPENDS blockforest field )
    waLBerla_execute_test( NAME ConfigFromPythonTest 
                           COMMAND $<TARGET_FILE:ConfigFromPythonTest> ${CMAKE_CURRENT_SOURCE_DIR}/ConfigFromPythonTest.py )
                           
    waLBerla_compile_test( FILES CallbackTest.cpp  DEPENDS blockforest field )
    waLBerla_execute_test( NAME CallbackTest
                          COMMAND $<TARGET_FILE:CallbackTest> ${CMAKE_CURRENT_SOURCE_DIR}/CallbackTest.py )
    set_target_properties(CallbackTest PROPERTIES CXX_VISIBILITY_PRESET hidden)

    waLBerla_compile_test( FILES FieldExportTest.cpp  DEPENDS blockforest field )
    waLBerla_execute_test( NAME FieldExportTest
                           COMMAND $<TARGET_FILE:FieldExportTest> ${CMAKE_CURRENT_SOURCE_DIR}/FieldExportTest.py )
    set_target_properties(FieldExportTest PROPERTIES CXX_VISIBILITY_PRESET hidden)

    add_test( NAME PythonBlockForestTest
          COMMAND python3 -m unittest discover waLBerla_tests/test_blockforest.py WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python/)
    SET_TESTS_PROPERTIES(PythonBlockForestTest
          PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/apps/pythonmodule:$PYTHONPATH")

    add_test( NAME PythonCoreTest
            COMMAND python3 -m unittest discover waLBerla_tests/test_core.py WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python/)
    SET_TESTS_PROPERTIES(PythonCoreTest
            PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/apps/pythonmodule:$PYTHONPATH")

    add_test( NAME PythonFieldTest
            COMMAND python3 -m unittest discover waLBerla_tests/test_field.py WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python/)
    SET_TESTS_PROPERTIES(PythonFieldTest
            PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/apps/pythonmodule:$PYTHONPATH")
endif()

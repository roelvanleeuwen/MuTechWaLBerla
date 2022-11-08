
waLBerla_link_files_to_builddir( *.prm )

if( WALBERLA_BUILD_WITH_CODEGEN )
    waLBerla_generate_target_from_python( NAME      RayleighBenardConvectionLatticeModelGeneration
            FILE      RayleighBenardConvectionLatticeModelGeneration.py
            OUT_FILES RBCLatticeModel.cpp RBCLatticeModel.h
                      RBCPackInfo.cpp RBCPackInfo.h )

    waLBerla_add_executable(NAME     RayleighBenardConvection
            FILES    RayleighBenardConvection.cpp
            DEPENDS  blockforest boundary core domain_decomposition field lbm
                     timeloop vtk RayleighBenardConvectionLatticeModelGeneration)
endif()

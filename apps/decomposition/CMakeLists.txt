waLBerla_link_files_to_builddir( *.cfg )
waLBerla_link_files_to_builddir( *.off )
waLBerla_add_executable ( NAME CubeDecomposition
                          FILES CubeDecomposition.cpp
                          DEPENDS blockforest mesh core pe vtk)

waLBerla_add_executable ( NAME ComplexDecomposition
                          FILES ComplexDecomposition.cpp
                          DEPENDS blockforest mesh core pe vtk)

waLBerla_add_executable ( NAME Armadillo
                          FILES Armadillo.cpp
                          DEPENDS blockforest mesh core pe vtk)


waLBerla_add_executable ( NAME Rhein
                          FILES Rhein.cpp
                          DEPENDS blockforest mesh core pe vtk)

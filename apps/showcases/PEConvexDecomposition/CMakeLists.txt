# CMake Lists for Armadillo Decomposition showcase
waLBerla_link_files_to_builddir( *.off )
waLBerla_add_executable ( NAME ArmadilloMeshDecomposition
                          FILES ArmadilloMeshDecomposition.cpp
                          DEPENDS blockforest mesh core pe vtk) 

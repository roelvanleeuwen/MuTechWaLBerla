waLBerla_link_files_to_builddir(*.prm)

waLBerla_add_executable(NAME SettlingSpheres
        FILES SettlingSpheres.cpp
        DEPENDS blockforest core mesa_pd vtk)

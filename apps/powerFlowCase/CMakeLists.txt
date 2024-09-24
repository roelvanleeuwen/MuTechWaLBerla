waLBerla_link_files_to_builddir( *.prm )

                                                            
waLBerla_add_executable ( NAME powerFlowCase
                            FILES PowerFlowCase.cpp
                            DEPENDS blockforest core field geometry timeloop gui )

                        
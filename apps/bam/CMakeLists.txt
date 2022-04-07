waLBerla_link_files_to_builddir( *.cfg )

#waLBerla_add_executable ( NAME MESA_PD_APP_Cohesion
#                          FILES Cohesion.cpp
#                          DEPENDS blockforest core mesa_pd vtk )
#
#waLBerla_add_executable ( NAME MESA_PD_APP_MinimalCohesion
#        FILES MinimalCohesion.cpp
#        DEPENDS blockforest core mesa_pd vtk )
#
#waLBerla_add_executable ( NAME MESA_PD_APP_MinimalCohesion_2
#        FILES MinimalCohesion_2.cpp
#        DEPENDS blockforest core mesa_pd vtk )
#
#waLBerla_add_executable ( NAME TractionTest
#        FILES TractionTest.cpp
#        DEPENDS blockforest core mesa_pd vtk )
#
#
#waLBerla_add_executable ( NAME SlidingTest
#        FILES SlidingTest.cpp
#        DEPENDS blockforest core mesa_pd vtk )
#
#waLBerla_add_executable ( NAME RollingTest
#        FILES RollingTest.cpp
#        DEPENDS blockforest core mesa_pd vtk )
#
#
#waLBerla_add_executable ( NAME SettlingParticlesWithCohesion
#        FILES SettlingParticlesWithCohesion.cpp
#        DEPENDS blockforest core mesa_pd vtk )
#
#
#waLBerla_add_executable ( NAME MESA_PD_APP_Bucket
#        FILES Bucket.cpp
#        DEPENDS blockforest core mesa_pd vtk )

#waLBerla_add_executable ( NAME MESA_PD_APP_BucketCohesion
#        FILES BucketCohesion.cpp
#        DEPENDS blockforest core mesa_pd vtk )

waLBerla_add_executable ( NAME MESA_PD_APP_CohBucket_settling
        FILES CohBucket_settling.cpp
        DEPENDS blockforest core mesa_pd vtk )


waLBerla_add_executable ( NAME MESA_PD_APP_act
        FILES CohBucket_act.cpp
        DEPENDS blockforest core mesa_pd vtk )

waLBerla_add_executable ( NAME MESA_PD_APP_allInOne
        FILES AllInOne.cpp
        DEPENDS blockforest core mesa_pd vtk )
        
 waLBerla_add_executable ( NAME MESA_PD_APP_BoxParticlesCohesion
        FILES BoxParticles_Cohesion.cpp
        DEPENDS blockforest core mesa_pd vtk )

waLBerla_add_executable ( NAME MESA_PD_APP_ParticleBed_Water
        FILES ParticleBed_Water.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd postprocessing timeloop vtk )



waLBerla_add_executable ( NAME MESA_PD_APP_One_Settling_Pierre
        FILES One_Particles_Settling_NO_Water_Pierre.cpp
        DEPENDS blockforest core mesa_pd lbm vtk )

waLBerla_add_executable ( NAME MESA_PD_APP_Two_ParticlesAndWater_Pierre
        FILES Two_Particles_and_Water_Pierre.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd postprocessing timeloop vtk )




waLBerla_add_executable ( NAME MESA_PD_APP_Three_Adding_Bucket_Pierre
        FILES Three_Adding_Bucket_Pierre.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd postprocessing timeloop vtk )


waLBerla_add_executable ( NAME MESA_PD_APP_Four_Adding_Box_Pierre
        FILES Four_Adding_Box_Pierre.cpp
        DEPENDS blockforest boundary core domain_decomposition field lbm lbm_mesapd_coupling mesa_pd postprocessing timeloop vtk )

waLBerla_add_executable ( NAME MESA_PD_APP_CohesionExperimentNormal
      FILES CohesionExperimentNormal.cpp
      DEPENDS blockforest core mesa_pd postprocessing vtk )

waLBerla_add_executable ( NAME MESA_PD_APP_TwoSpheresOnOne
      FILES TwoSpheresOnOne.cpp
      DEPENDS blockforest core domain_decomposition mesa_pd vtk )

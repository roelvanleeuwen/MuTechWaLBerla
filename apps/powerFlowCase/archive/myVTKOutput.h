#include "blockforest/all.h"

#include "core/all.h"
#include "core/debug/Debug.h"

#include "domain_decomposition/all.h"

#include "field/AddToStorage.h"
#include "field/FlagField.h"
#include "field/GhostLayerField.h"
#include "field/all.h"
#include "field/communication/PackInfo.h"
#include "field/iterators/FieldPointer.h"
#include "field/iterators/IteratorMacros.h"

#include "geometry/all.h"

#include "gui/all.h"

#include "lbm/all.h"
#include "lbm/blockforest/communication/SimpleCommunication.h"
#include "lbm/lattice_model/SmagorinskyLES.h"

#include "mesh/blockforest/BlockExclusion.h"
#include "mesh/blockforest/BlockForestInitialization.h"
#include "mesh/blockforest/BlockWorkloadMemory.h"
#include "mesh/blockforest/RefinementSelection.h"
#include "mesh/boundary/BoundaryInfo.h"
#include "mesh/boundary/BoundaryLocation.h"
#include "mesh/boundary/BoundaryLocationFunction.h"
#include "mesh/boundary/BoundarySetup.h"
#include "mesh/boundary/BoundaryUIDFaceDataSource.h"
#include "mesh/boundary/ColorToBoundaryMapper.h"

#include "timeloop/SweepTimeloop.h"
#include "timeloop/all.h"

#include "vtk/ChainedFilter.h"

#include "mesh_common/DistanceComputations.h"
#include "mesh_common/DistanceFunction.h"
#include "mesh_common/MatrixVectorOperations.h"
#include "mesh_common/MeshIO.h"
#include "mesh_common/MeshOperations.h"
#include "mesh_common/TriangleMeshes.h"
#include "mesh_common/distance_octree/DistanceOctree.h"
#include "mesh_common/vtk/CommonDataSources.h"
#include "mesh_common/vtk/VTKMeshWriter.h"

#include <vector>

namespace walberla
{

/////////
// VTK //
/////////

template< typename LatticeModel_T >
class MyVTKOutput
{
 public:
   MyVTKOutput(const ConstBlockDataID& pdfField, const ConstBlockDataID& omegaField, const ConstBlockDataID& flagField,
               const vtk::VTKOutput::BeforeFunction& pdfGhostLayerSync)
      : pdfField_(pdfField), omegaField_(omegaField), flagField_(flagField), pdfGhostLayerSync_(pdfGhostLayerSync)
   {}

   void operator()(std::vector< shared_ptr< vtk::BlockCellDataWriterInterface > >& writers,
                   std::map< std::string, vtk::VTKOutput::CellFilter >& filters,
                   std::map< std::string, vtk::VTKOutput::BeforeFunction >& beforeFunctions);

 private:
   const ConstBlockDataID pdfField_;
   const ConstBlockDataID omegaField_;
   const ConstBlockDataID flagField_;

   vtk::VTKOutput::BeforeFunction pdfGhostLayerSync_;

}; // class MyVTKOutput

template< typename LatticeModel_T >
void MyVTKOutput< LatticeModel_T >::operator()(std::vector< shared_ptr< vtk::BlockCellDataWriterInterface > >& writers,
                                               std::map< std::string, vtk::VTKOutput::CellFilter >& filters,
                                               std::map< std::string, vtk::VTKOutput::BeforeFunction >& beforeFunctions)
{
   // block data writers

   writers.push_back(make_shared< lbm::VelocitySIVTKWriter< LatticeModel_T, float > >(pdfField_, units_.xSI, units_.tSI,
                                                                                      "Velocity"));
   writers.push_back(
      make_shared< lbm::DensitySIVTKWriter< LatticeModel_T, float > >(pdfField_, units_.rhoSI, "Density"));
   writers.push_back(make_shared< lbm::VTKWriter< ScalarField_T > >(omegaField_, "OmegaField"));
   writers.push_back(make_shared< field::VTKWriter< FlagField_T > >(flagField_, "FlagField"));

   // cell filters

   field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagField_);
   fluidFilter.addFlag(Fluid_Flag);
   filters["FluidFilter"] = fluidFilter;

   field::FlagFieldCellFilter< FlagField_T > obstacleFilter(flagField_);
   obstacleFilter.addFlag(NoSlip_Flag);
   obstacleFilter.addFlag(Obstacle_Flag);
   obstacleFilter.addFlag(Curved_Flag);
   obstacleFilter.addFlag(UBB_Flag);
   obstacleFilter.addFlag(PressureOutlet_Flag);
   obstacleFilter.addFlag(Outlet21_Flag);
   obstacleFilter.addFlag(Outlet43_Flag);
   filters["ObstacleFilter"] = obstacleFilter;

   // before functions

   beforeFunctions["PDFGhostLayerSync"] = pdfGhostLayerSync_;
}

} // namespace walberla
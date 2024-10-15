#include "blockforest/all.h"

#include "core/Environment.h"

#include <cmath>
#include <fstream>
#include <iostream>
using namespace std;
using namespace walberla;

namespace walberla
{

struct Units
{
   friend std::ostream& operator<<(std::ostream& os, const Units& units);
   // Physical units
   real_t xSI; // input from user (choice: lattice spacing of the coarses refinement level)
   real_t tSI;
   real_t speedSI;
   real_t kinViscositySI;
   real_t temperatureSI;
   real_t rhoSI;
   real_t massSI;
   real_t speedOfSoundSI;
   real_t mach;
   real_t Re;

   // Unitless units
   real_t rhoUnitless   = 1.0; // density unit FREE PARAMETER
   real_t speedUnitless = 1.0; // velocity unit FREE PARAMETER

   real_t kinViscosityUnitless;
   real_t speedOfSoundUnitless;

   // Lattice units
   real_t omega;               // setting of the relaxation parameter (on the coarsest level)
   uint_t omegaLevel;          // level where omega is defined (choice: coarsest level)
   real_t rhoLU         = 1.0; // density unit FREE PARAMETER
   real_t temperatureLU = 1.0; // temperatureSI unit FREE PARAMETER

   real_t massLU;
   real_t speedLU;
   real_t kinViscosityLU;
   real_t tLU;
   real_t xLU;
   real_t speedOfSoundLU;
   real_t pseudoSpeedOfSoundLU = 1 / std::sqrt(3.0);
   real_t machLU; // Mach number in lattice units FREE PARAMETER set around 0.1 for incompressible flow
   real_t mach_check;
};

Units convertToLatticeUnits(const Units& inputUnits, const bool unitsWriter)
{
   Units units = inputUnits;
   // Calculated physical units
   units.massSI         = units.rhoSI * std::pow(units.xSI, 3);
   units.tSI            = units.speedUnitless / units.speedSI * units.xSI;
   units.speedOfSoundSI = std::sqrt(1.4 * 287.15 * units.temperatureSI);

   // Calculated unitless units
   units.mach                 = units.speedSI / units.speedOfSoundSI;
   units.Re                   = units.speedSI * units.xSI / units.kinViscositySI;
   units.kinViscosityUnitless = units.kinViscositySI * units.tSI / (units.xSI * units.xSI);
   units.speedOfSoundUnitless = units.speedOfSoundSI * units.tSI / units.xSI;

   // Calculated lattice units
   units.massLU         = units.rhoLU * std::pow(units.xSI, 3) / units.rhoUnitless;
   units.speedLU        = units.machLU * units.pseudoSpeedOfSoundLU;
   units.kinViscosityLU = (1 / units.omega - 0.5) / std::pow(units.pseudoSpeedOfSoundLU, 2);
   units.tLU            = units.kinViscosityLU / units.kinViscosityUnitless * (units.speedUnitless / units.speedLU) *
               (units.speedUnitless / units.speedLU);
   units.xLU            = units.kinViscosityLU / units.kinViscosityUnitless * units.speedUnitless / units.speedLU;
   units.speedOfSoundLU = units.speedOfSoundUnitless * units.xLU / units.tLU;
   // units.MachLU         = units.speedLU / units.pseudoSpeedOfSoundLU;
   units.mach_check = units.speedLU / units.speedOfSoundLU;

   if (unitsWriter)
   {
      // Write the simulation setup to a file
      std::ofstream outFile("Units.txt");
      if (outFile.is_open())
      {
         outFile << units;
         outFile.close();
         WALBERLA_LOG_INFO_ON_ROOT("Units written to file successfully")
      }
      else { WALBERLA_LOG_INFO_ON_ROOT("Failed to open file for writing Units") }
   };

   return units;
}
std::ostream& operator<<(std::ostream& os, const Units& units)
{
   os << "================= Units ===============: \n"
      << "User input: \n"
      << "xSI: " << units.xSI << " m \n"
      << "speedSI: " << units.speedSI << " m/s \n"
      << "kinViscositySI: " << units.kinViscositySI << " m2/s \n"
      << "rhoSI: " << units.rhoSI << " kg/m3\n"
      << "temperatureSI: " << units.temperatureSI << " K \n"
      << "massSI: " << units.massSI << " kg \n"
      << "omega: " << units.omega << " - \n"
      << "MachLU: " << units.machLU << " - \n"
      << "  \n"
      << "Default units: \n"
      << "rhoUnitless: " << units.rhoUnitless << " - \n"
      << "speedUnitless: " << units.speedUnitless << " - \n"
      << "rhoLU: " << units.rhoLU << " mu/lu3\n"
      << "temperatureLU: " << units.temperatureLU << " tu \n"
      << " \n"
      << "Physical units: \n"
      << "tSI: " << units.tSI << " s \n"
      << "speedOfSoundSI: " << units.speedOfSoundSI << " m/s \n"
      << "massSI: " << units.massSI << " kg \n"
      << " \n"
      << "Calculated unitless units: \n"
      << "mach: " << units.mach << " - \n"
      << "Re: " << units.Re << " - \n"
      << "kinViscosityUnitless: " << units.kinViscosityUnitless << " - \n"
      << "speedOfSoundUnitless: " << units.speedOfSoundUnitless << " - \n"
      << " \n"
      << "Calculated lattice units: \n"
      << "massLU: " << units.massLU << " mu \n"
      << "speedLU: " << units.speedLU << " lu/ts\n"
      << "kinViscosityLU: " << units.kinViscosityLU << " lu2/ts\n"
      << "tLU: " << units.tLU << " ts\n"
      << "xLU: " << units.xLU << " lu\n"
      << "speedOfSoundLU: " << units.speedOfSoundLU << " lu/ts\n"
      << "pseudoSpeedOfSoundLU: " << units.pseudoSpeedOfSoundLU << " lu/ts\n"
      << "mach_check: " << units.mach_check << " - \n"
      << " \n"
      << "Conversion factors: \n"
      << " tLU/tSI: " << units.tLU / units.tSI << " ts/s \n"
      << " xLU/xSI: " << units.xLU / units.xSI << " lu/m \n"
      << " massLU/massSI: " << units.massLU / units.massSI << " mu/kg \n";
   return os;
}

} // namespace walberla
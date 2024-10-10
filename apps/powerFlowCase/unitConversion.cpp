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
   real_t x; // input from user (choice: lattice spacing of the coarses refinement level)
   real_t t;
   real_t speed;
   real_t kinViscosity;
   real_t temperature;
   real_t rho;
   real_t mass;
   real_t speedOfSound;
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
   real_t temperatureLU = 1.0; // temperature unit FREE PARAMETER

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
   units.mass         = units.rho * std::pow(units.x, 3);
   units.t            = units.speedUnitless / units.speed * units.x;
   units.speedOfSound = std::sqrt(1.4 * 287.15 * units.temperature);

   // Calculated unitless units
   units.mach                 = units.speed / units.speedOfSound;
   units.Re                   = units.speed * units.x / units.kinViscosity;
   units.kinViscosityUnitless = units.kinViscosity * units.t / (units.x * units.x);
   units.speedOfSoundUnitless = units.speedOfSound * units.t / units.x;

   // Calculated lattice units
   units.massLU         = units.rhoLU * std::pow(units.x, 3) / units.rhoUnitless;
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
      << "x: " << units.x << " m \n"
      << "speed: " << units.speed << " m/s \n"
      << "kinViscosity: " << units.kinViscosity << " m2/s \n"
      << "rho: " << units.rho << " kg/m3\n"
      << "temperature: " << units.temperature << " K \n"
      << "mass: " << units.mass << " kg \n"
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
      << "t: " << units.t << " s \n"
      << "speedOfSound: " << units.speedOfSound << " m/s \n"
      << "mass: " << units.mass << " kg \n"
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
      << " tLU/t: " << units.tLU / units.t << " ts/s \n"
      << " xLU/x: " << units.xLU / units.x << " lu/m \n"
      << " massLU/mass: " << units.massLU / units.mass << " mu/kg \n";
   return os;
}

} // namespace walberla
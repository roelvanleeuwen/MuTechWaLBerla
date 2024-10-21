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
   // Physical units
   real_t xSI; // input from user (choice: lattice spacing of the coarses refinement level)
   real_t tSI;
   real_t rhoSI;
   real_t massSI;
   real_t temperatureSI;
   real_t speedSI;
   real_t speedOfSoundSI;
   real_t kinViscositySI;
   real_t MachSI;
   real_t ReSI;

   // Lattice units
   real_t xLU           = 1.0; // length unit FREE PARAMETER set to unitity
   real_t tLU           = 1.0; // time unit FREE PARAMETER set to unitity
   real_t rhoLU         = 1.0; // density unit FREE PARAMETER set to unitity
   real_t massLU        = rhoLU * std::pow(xLU, 3);
   real_t temperatureLU = 1.0; // temperatureSI unit FREE PARAMETER
   real_t speedLU;
   real_t kinViscosityLU;
   real_t speedOfSoundLU;
   real_t pseudoSpeedOfSoundLU = 1 / std::sqrt(3.0);
   real_t MachLU; // Mach number in lattice units FREE PARAMETER set around 0.1 for incompressible flow
   real_t ReLU;
   real_t omegaLUTheory;  // theoretical omega value from acoustic scaling
   real_t omegaEffective; // effective relaxation parameter
   uint_t omegaLevel;     // level where omega is defined (choice: coarsest level)

   // Conversion factors
   real_t thetaMass;
   real_t thetaTime;
   real_t thetaLength;
   real_t thetaTemperature;
   real_t thetaSpeed;

   // Simulation conditions
   Vector3 < real_t > initialVelocityLU;
   Vector3 < real_t > flowVelocityLU;

   // Define the operator<< for Units
   friend std::ostream& operator<<(std::ostream& os, const Units& units)
   {
      os << "================= Units ===============: \n"
         << "User input: \n"
         << "  xSI: " << units.xSI << " m \n"
         << "  tSI: " << units.tSI << " s \n"
         << "  rhoSI: " << units.rhoSI << " kg/m3\n"
         << "  massSI: " << units.massSI << " kg \n"
         << "  temperatureSI: " << units.temperatureSI << " K \n"
         << "  speedSI: " << units.speedSI << " m/s \n"
         << "  speedOfSoundSI: " << units.speedOfSoundSI << " m/s \n"
         << "  kinViscositySI: " << units.kinViscositySI << " m2/s \n"
         << " \n"
         << "Lattice units: \n"
         << "  xLU: " << units.xLU << " lu\n"
         << "  tLU: " << units.tLU << " ts\n"
         << "  rhoLU: " << units.rhoLU << " - \n"
         << "  massLU: " << units.massLU << " mu \n"
         << "  temperatureLU: " << units.temperatureLU << " - \n"
         << "  speedLU: " << units.speedLU << " lu/ts\n"
         << "  kinViscosityLU: " << units.kinViscosityLU << " lu2/ts\n"
         << "  speedOfSoundLU: " << units.speedOfSoundLU << " lu/ts\n"
         << "  pseudoSpeedOfSoundLU: " << units.pseudoSpeedOfSoundLU << " lu/ts\n"
         << "  MachLU: " << units.MachLU << " - \n"
         << "  ReLU: " << units.ReLU << " - \n"
         << " \n"
         << "Conversion factors: \n"
         << "  thetaMass: " << units.thetaMass << " mu/kg \n"
         << "  thetaTime: " << units.thetaTime << " ts/s \n"
         << "  thetaLength: " << units.thetaLength << " lu/m \n"
         << "  thetaTemperature: " << units.thetaTemperature << " - \n"
         << " \n"
         << "Other parameters: \n"
         << "  omegaLUTheory: " << units.omegaLUTheory << " - \n"
         << "  omegaEffective: " << units.omegaEffective << " - \n"
         << "  omegaLevel: " << units.omegaLevel << " - \n"
         << " \n"
         << "Initial conditions and boundary conditions: \n"
         << "  initialVelocityLU: " << units.initialVelocityLU << " lu/ts \n"
         << "  flowVelocityLU: " << units.flowVelocityLU << " lu/ts \n"
         << " \n";

         // Add other members as needed
         return os;
   }
};

Units convertToLatticeUnitsAcousticScaling(Units& units)
{
   // Input units:
   //    xSI
   //    speedSI
   //    kinViscositySI
   //    rhoSI
   //    temperatureSI
   //    omega_chosen
   //    omega_level

   // Calculated physical units
   units.massSI         = units.rhoSI * std::pow(units.xSI, 3);
   units.speedOfSoundSI = std::sqrt(1.4 * 287.15 * units.temperatureSI);
   units.MachSI         = units.speedSI / units.speedOfSoundSI;

   // Acoustic scaling
   units.MachLU  = units.MachSI;
   units.tSI     = units.xSI * units.pseudoSpeedOfSoundLU / units.speedOfSoundSI;
   units.speedLU = units.MachLU * units.pseudoSpeedOfSoundLU;

   units.ReSI           = units.speedSI * units.xSI / units.kinViscositySI;
   units.kinViscosityLU = units.kinViscositySI * units.tSI / (std::pow(units.xSI, 2.0));
   units.ReLU           = units.speedLU * units.xLU / units.kinViscosityLU;
   real_t tauLU         = 0.5 + units.kinViscosityLU / std::pow(units.pseudoSpeedOfSoundLU, 2);
   units.omegaLUTheory  = 1.0 / (tauLU);

   // Scaling factors
   units.thetaMass        = units.massLU / units.massSI;
   units.thetaTime        = units.tLU / units.tSI;
   units.thetaLength      = units.xLU / units.xSI;
   units.thetaTemperature = units.temperatureLU / units.temperatureSI;
   units.thetaSpeed       = units.thetaLength / units.thetaTime;

   return units;
}

} // namespace walberla
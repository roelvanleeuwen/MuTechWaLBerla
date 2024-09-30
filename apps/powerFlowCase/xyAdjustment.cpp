/*
This file is to calculate the required xyAdjustment when the number of blocks in the x and z dimension are maximised. 
    The maximisation is succeeded when the number of cells in the x and z direction in one block is just above 16. 
*/

#include <iostream>
#include <cmath>
using namespace std;
using namespace walberla;

struct AdjustmentResult {
    walberla::real_t xyAdjustment;
    walberla::uint_t nBlocks_x;
    walberla::uint_t cellsPerBlock_x;
};

AdjustmentResult xyAdjustment(const walberla::real_t xSize, const walberla::real_t decreasePowerFlowDomainFactor, const walberla::real_t dx)
{
    walberla::real_t nBlocks_x = 3;
    walberla::real_t nBlocks_x_remember = nBlocks_x;

    walberla::real_t cellsPerBlock_x = xSize * decreasePowerFlowDomainFactor / (dx * nBlocks_x);
    walberla::real_t cellsPerBlock_x_remember = cellsPerBlock_x;

    if (cellsPerBlock_x < 16.0)
    {
        std::cout << "The number of cells in the x and z direction are below 16. This is not allowed. The decreasePowerFlowDomainFactor should be "
              << "increased to start with more than 16 cells in the x and z dimension." << std::endl;
    }
    while (cellsPerBlock_x > 16.0)
    {
        nBlocks_x_remember = nBlocks_x;
        nBlocks_x = nBlocks_x+2;
        
        cellsPerBlock_x_remember = cellsPerBlock_x;
        cellsPerBlock_x = xSize * decreasePowerFlowDomainFactor / (dx * nBlocks_x);

    }
    // Round the number of cellsPerBlock_x_remember to the nearest integer
    cellsPerBlock_x = std::round(cellsPerBlock_x_remember);
    nBlocks_x = nBlocks_x_remember;

    walberla::real_t xyAdjustment = (nBlocks_x * cellsPerBlock_x * dx) / (xSize * decreasePowerFlowDomainFactor);
    
    return {xyAdjustment, static_cast<walberla::uint_t>(nBlocks_x), static_cast<walberla::uint_t>(cellsPerBlock_x)};
}
/*
void testOne ()
{
    // Test case 1
    const walberla::real_t xSize = 20;
    const walberla::real_t decreasePowerFlowDomainFactor = 0.013;
    const walberla::real_t dx = 0.0051015625;
    auto result = xyAdjustment(xSize, decreasePowerFlowDomainFactor, dx);
    std::cout << "Test case 1: " << std::endl;
    std::cout << "xyAdjustment: " << result.xyAdjustment << std::endl;
    std::cout << "nBlocks_x: " << result.nBlocks_x << std::endl;
    std::cout << "cellsPerBlock_x: " << result.cellsPerBlock_x << std::endl;
}

void testTwo ()
{
    // Test case 1
    const walberla::real_t xSize = 20;
    const walberla::real_t decreasePowerFlowDomainFactor = 0.02;
    const walberla::real_t dx = 0.0051015625;
    auto result = xyAdjustment(xSize, decreasePowerFlowDomainFactor, dx);
    std::cout << "Test case 2: " << std::endl;
    std::cout << "xyAdjustment: " << result.xyAdjustment << std::endl;
    std::cout << "nBlocks_x: " << result.nBlocks_x << std::endl;
    std::cout << "cellsPerBlock_x: " << result.cellsPerBlock_x << std::endl;
}

void testThree ()
{
    // Test case 1
    const walberla::real_t xSize = 20;
    const walberla::real_t decreasePowerFlowDomainFactor = 0.5;
    const walberla::real_t dx = 0.0051015625;
    auto result = xyAdjustment(xSize, decreasePowerFlowDomainFactor, dx);
    std::cout << "Test case 3: " << std::endl;
    std::cout << "xyAdjustment: " << result.xyAdjustment << std::endl;
    std::cout << "nBlocks_x: " << result.nBlocks_x << std::endl;
    std::cout << "cellsPerBlock_x: " << result.cellsPerBlock_x << std::endl;
}

int main() {
    // Run the test
    testOne();
    testTwo();
    testThree();
    return 0;
}
*/


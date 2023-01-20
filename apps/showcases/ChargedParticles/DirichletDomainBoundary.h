#ifndef WALBERLA_DIRICHLETDOMAINBOUNDARY_H
#define WALBERLA_DIRICHLETDOMAINBOUNDARY_H

namespace walberla
{
template< typename PdeField >
class DirichletDomainBoundary
{
 public:
   DirichletDomainBoundary(StructuredBlockStorage& blocks, const BlockDataID& fieldId)
      : blocks_(blocks), fieldId_(fieldId)
   {
      for (uint_t i = 0; i != stencil::D3Q6::Size; ++i)
      {
         includeBoundary_[i] = true;
         value_              = real_t(0);
      }
   }

   void includeBoundary(const stencil::Direction& direction) { includeBoundary_[stencil::D3Q6::idx[direction]] = true; }
   void excludeBoundary(const stencil::Direction& direction)
   {
      includeBoundary_[stencil::D3Q6::idx[direction]] = false;
   }

   void setValue(const real_t value) { value_ = value; }

   void operator()();

 protected:
   // first-order dirichlet boundary conditions with scalar value
   void apply(PdeField* p, const CellInterval& interval, const cell_idx_t cx, const cell_idx_t cy, const cell_idx_t cz,
              const real_t value) const;

   StructuredBlockStorage& blocks_;
   BlockDataID fieldId_;

   bool includeBoundary_[stencil::D3Q6::Size];

   real_t value_;

}; // class DirichletDomainBoundary

template< typename PdeField >
void DirichletDomainBoundary< PdeField >::operator()()
{
   for (auto block = blocks_.begin(); block != blocks_.end(); ++block)
   {
      PdeField* p = block->template getData< PdeField >(fieldId_);

      if (includeBoundary_[stencil::D3Q6::idx[stencil::W]] && blocks_.atDomainXMinBorder(*block))
      {
         apply(p,
               CellInterval(
                  cell_idx_t(-1), cell_idx_t(0)                         , cell_idx_t(0),
                  cell_idx_t(-1), cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(1), cell_idx_t(0), cell_idx_t(0), value_);
      }
      if (includeBoundary_[stencil::D3Q6::idx[stencil::E]] && blocks_.atDomainXMaxBorder(*block))
      {
         apply(p,
               CellInterval(
                  cell_idx_c(p->xSize()), cell_idx_t(0)                         , cell_idx_t(0),
                  cell_idx_c(p->xSize()), cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(-1), cell_idx_t(0), cell_idx_t(0), value_);
      }

      if (includeBoundary_[stencil::D3Q6::idx[stencil::S]] && blocks_.atDomainYMinBorder(*block))
      {
         apply(p,
               CellInterval(
                  cell_idx_t(0)                         , cell_idx_t(-1), cell_idx_t(0),
                  cell_idx_c(p->xSize()) - cell_idx_t(1), cell_idx_t(-1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(0), cell_idx_t(1), cell_idx_t(0), value_);
      }
      if (includeBoundary_[stencil::D3Q6::idx[stencil::N]] && blocks_.atDomainYMaxBorder(*block))
      {
         apply(p,
               CellInterval(
                  cell_idx_t(0)                         , cell_idx_c(p->ySize()), cell_idx_t(0),
                  cell_idx_c(p->xSize()) - cell_idx_t(1), cell_idx_c(p->ySize()), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(0), cell_idx_t(-1), cell_idx_t(0), value_);
      }

      if (includeBoundary_[stencil::D3Q6::idx[stencil::B]] && blocks_.atDomainZMinBorder(*block))
      {
         apply(p,
               CellInterval(
                  cell_idx_t(0)                         , cell_idx_t(0)                         , cell_idx_t(-1),
                  cell_idx_c(p->xSize()) - cell_idx_t(1), cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_t(-1)),
               cell_idx_t(0), cell_idx_t(0), cell_idx_t(1), value_);
      }
      if (includeBoundary_[stencil::D3Q6::idx[stencil::T]] && blocks_.atDomainZMaxBorder(*block))
      {
         apply(p,
               CellInterval(
                  cell_idx_t(0)                         , cell_idx_t(0)                         , cell_idx_c(p->zSize()),
                  cell_idx_c(p->xSize()) - cell_idx_t(1), cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize())),
               cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1), value_);
      }
   }
}

template< typename PdeField >
void DirichletDomainBoundary< PdeField >::apply(PdeField* p, const CellInterval& interval, const cell_idx_t cx,
                                                const cell_idx_t cy, const cell_idx_t cz, const real_t value) const
{
   WALBERLA_FOR_ALL_CELLS_IN_INTERVAL_XYZ(interval,
                                          p->get(x, y, z) = real_c(2) * value - p->get(x + cx, y + cy, z + cz);)
}

} // namespace walberla

#endif // WALBERLA_DIRICHLETDOMAINBOUNDARY_H

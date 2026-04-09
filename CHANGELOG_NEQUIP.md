
# Changelog for NequIP integration with PyRAI2MD

## [04-09-2026] NequIP Adaptive Sampling Support
### Added
- Support for **NequIP-NAC in PyRAI2MD adaptive sampling**.
- Adaptive loop now supports **no-seed startup** for NequIP: the first converged QC batch is used to initialize training data.
- Adaptive NequIP flow uses ensemble uncertainty while keeping NequIP training/retraining outside PyRAI2MD.

### Changed
- NequIP adaptive data update logic now safely handles empty QC harvests (no-op when no converged frames are returned).

### Deprecated / Compatibility
- Legacy keys under `&NEQUIP_EG` / `&NEQUIP_NAC` (`natom`, `model_path`, `gpu`) are now accepted as a **compatibility shim** but **ignored** at runtime.
- A warning is printed to guide migration to canonical `&NEQUIP` keys (e.g., `modeldir`, `gpu`), and atom count is auto-detected from structures.

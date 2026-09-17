# Verified legacy encoding assets

These small files must ship with source checkouts; they are not model weights.
`registry.json` binds the M0 model/EMA tensor fingerprint to `m0_pca16_v1.pt` and
its canonical encoding-state hash. See [provenance and schema](../../../docs/type_encoding.md).
Do not use this table as the default for fresh training or unrelated PCA models.

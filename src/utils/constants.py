DATASET_NMAX_DEFAULTS = {
    "mp20": 20,
    "mpts_52": 52,
    "perov_5": 5,
    "alex_mp20": 20,
}

AA_TARGET_ORDER = ("coords", "lattice")
DIAGNOSTIC_SECTION_KEYS = (
    "struct_min_dist_frac_lt_cutoff",
    "struct_volume_frac_lt_cutoff",
    "lattice_length_frac_lt_1",
    "comp_missing_ox_frac",
    "invalid_atom_frac",
    "num_elems_frac_single",
)

# Backward-compatible alias for older imports during the refactor.
_DIAGNOSTIC_SECTION_KEYS = DIAGNOSTIC_SECTION_KEYS

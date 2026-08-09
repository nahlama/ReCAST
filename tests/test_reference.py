from recast_surv.reference import harmonize_cell_state


def test_gse154763_myeloid_state_harmonization():
    assert harmonize_cell_state("M01_Mast_KIT") == "Mast"
    assert harmonize_cell_state("M03_cDC1_CLEC9A") == "DC"
    assert harmonize_cell_state("M07_Mono_CD16") == "Mono/Macro"
    assert harmonize_cell_state("M10_Macro_C1QC") == "Mono/Macro"
    assert harmonize_cell_state("Malignant") == "Malignant"

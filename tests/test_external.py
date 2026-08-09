import pandas as pd

from recast_surv.external import (
    build_sequence_crosswalk,
    external_clinical_features,
    parse_external_clinical,
)


def test_sequence_crosswalk_requires_unique_exact_sequence():
    feature = pd.DataFrame(
        {"id": ["1", "2", "3"], "sequence": ["ACGT", "TTAA", "TTAA"]}
    )
    probe = pd.DataFrame(
        {
            "id": ["probe-a", "probe-b"],
            "spot_id": ["GENE1", "GENE2"],
            "sequence": ["acgt", "ttaa"],
        }
    )
    crosswalk, summary = build_sequence_crosswalk(feature, probe)
    assert crosswalk.loc[crosswalk["feature_id"].eq("1"), "unique_sequence_match"].item()
    assert not crosswalk.loc[crosswalk["feature_id"].eq("2"), "unique_sequence_match"].item()
    assert summary["feature_rows_with_unique_match"] == 1


def test_external_clinical_parser_keeps_tumor_and_converts_time(tmp_path):
    metadata = pd.DataFrame(
        {
            "accession": ["GSE1", "GSE1"],
            "gsm_id": ["GSM1", "GSM2"],
            "sample_type_inferred": ["Tumor", "Normal"],
            "histology_inferred": ["ESCC", "ESCC"],
            "characteristics_text": [
                "patient id: ec1 | age: 60 | Sex: male | tnm stage: III | "
                "death at fu: yes | survival time(months): 12",
                "patient id: ec1 | age: 60 | Sex: male | tnm stage: III | "
                "death at fu: yes | survival time(months): 12",
            ],
        }
    )
    path = tmp_path / "metadata.csv"
    metadata.to_csv(path, index=False)
    result = parse_external_clinical(path)
    assert len(result) == 1
    assert result.loc[0, "stage_numeric"] == 3
    assert result.loc[0, "event"] == 1
    assert result.loc[0, "time_days"] == 365.25


def test_external_clinical_features_match_locked_encoding():
    clinical = pd.DataFrame(
        {
            "sample_id": ["GSM1"],
            "age": [60.0],
            "sex": ["male"],
            "histology": ["ESCC"],
            "stage_numeric": [3],
        }
    )
    result = external_clinical_features(clinical)
    assert result.loc["GSM1", "clinical__age_years"] == 60.0
    assert result.loc["GSM1", "clinical__sex_male"] == 1.0
    assert result.loc["GSM1", "clinical__histology_ESCC"] == 1.0
    assert result.loc["GSM1", "clinical__stage_III"] == 1.0
    assert result.loc["GSM1", "clinical__stage_UNKNOWN"] == 0.0

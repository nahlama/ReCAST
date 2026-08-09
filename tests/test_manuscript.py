import pandas as pd

from recast_surv.manuscript import cohort_characteristics


def test_cohort_characteristics_reports_counts_and_percentages():
    tcga = pd.DataFrame(
        {
            "event": [1, 0],
            "age_years": [50, 70],
            "sex_male": [1, 0],
            "histology": ["ESCC", "EAC"],
            "stage_group": ["II", None],
            "time_days": [100, 300],
        }
    )
    external = pd.DataFrame(
        {
            "event": [1],
            "age": [60],
            "sex": ["male"],
            "histology": ["ESCC"],
            "stage": ["III"],
            "time_days": [200],
        }
    )
    result = cohort_characteristics(tcga, external).set_index("characteristic")
    assert result.loc["Patients, n", "TCGA_ESCA_development"] == "2"
    assert result.loc["Deaths, n (%)", "TCGA_ESCA_development"] == "1 (50.0%)"
    assert result.loc["Stage UNKNOWN, n (%)", "TCGA_ESCA_development"] == "1 (50.0%)"

import gzip

import pandas as pd

from recast_surv.external_reannotation import map_probe_sequences_to_gencode


def test_exact_probe_mapping_requires_one_gene(tmp_path):
    crosswalk = pd.DataFrame(
        {
            "feature_id": ["1", "2", "3"],
            "sequence_key": ["ACGTACGTACGTACGTACGTACGTA", "TTTTCCCCAAAAGGGGTTTTCCCCA", "ACGTACGTACGTACGTACGTACGTA"],
            "feature_sequence_count": [1, 1, 2],
        }
    )
    crosswalk_path = tmp_path / "crosswalk.csv.gz"
    crosswalk.to_csv(crosswalk_path, index=False)
    fasta = tmp_path / "transcripts.fa.gz"
    with gzip.open(fasta, "wt") as handle:
        handle.write(">ENST1|ENSG1.1|x|x|T1|GENE1|\nNNNACGTACGTACGTACGTACGTACGTANNN\n")
        handle.write(">ENST2|ENSG2.1|x|x|T2|GENE2|\nNNNTTTTCCCCAAAAGGGGTTTTCCCCANNN\n")
        handle.write(">ENST3|ENSG3.1|x|x|T3|GENE3|\nNNNTTTTCCCCAAAAGGGGTTTTCCCCANNN\n")
    mapping, summary = map_probe_sequences_to_gencode(crosswalk_path, fasta, tmp_path / "out")
    status = mapping.set_index("feature_id")["mapping_status"]
    assert status["1"] == "unique_gene_exact_match"
    assert status["2"] == "ambiguous_gene_exact_match"
    assert status["3"] == "duplicated_feature_sequence"
    assert summary["transcripts_scanned"] == 3

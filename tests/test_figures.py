import matplotlib as mpl

from recast_surv.figures import configure_manuscript_style
from recast_surv.supplementary_figures import MODEL_LABELS


def test_manuscript_svg_style_is_editable_bold_times_new_roman():
    configure_manuscript_style()
    assert mpl.rcParams["svg.fonttype"] == "none"
    assert mpl.rcParams["font.weight"] == "bold"
    assert mpl.rcParams["font.serif"][0] == "Times New Roman"


def test_supplementary_model_labels_cover_locked_models():
    assert {"elastic_net_cox", "survivalpfn", "survpfn"}.issubset(MODEL_LABELS)

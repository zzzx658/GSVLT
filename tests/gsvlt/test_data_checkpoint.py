from dataclasses import replace

import torch

from yolox.tracker.gsvlt.checkpoint import load_checkpoint, save_checkpoint
from yolox.tracker.gsvlt.config import GSVLTConfig
from yolox.tracker.gsvlt.data import GSVLTMOTDataset
from yolox.tracker.gsvlt.gmvi import GMVI
from yolox.tracker.gsvlt.gva import GVA


def test_checkpoint_round_trip(tmp_path):
    config = replace(
        GSVLTConfig(),
        history_size=5,
        hidden_dim=16,
        positional_dim=16,
        attention_heads=4,
        attention_layers=1,
        max_group_size=4,
    )
    torch.manual_seed(7)
    gva = GVA(config)
    gmvi = GMVI(config)
    path = tmp_path / "model.pt"
    save_checkpoint(path, config, gva, gmvi, {"dataset": "test"})
    loaded_config, loaded_gva, loaded_gmvi, metadata = load_checkpoint(
        path, torch.device("cpu"), config
    )
    assert loaded_config == config
    assert metadata == {"dataset": "test"}
    for expected, actual in zip(gva.parameters(), loaded_gva.parameters()):
        assert torch.equal(expected, actual)
    for expected, actual in zip(gmvi.parameters(), loaded_gmvi.parameters()):
        assert torch.equal(expected, actual)


def test_mot_dataset_builds_padded_training_sample(tmp_path):
    sequence = tmp_path / "MOT17-TEST"
    (sequence / "gt").mkdir(parents=True)
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT17-TEST\nimWidth=200\nimHeight=100\nframeRate=30\n",
        encoding="utf-8",
    )
    rows = []
    for frame in range(1, 11):
        rows.append("{0},1,{0},20,10,20,1,1,1\n".format(frame))
        rows.append("{0},2,{1},20,10,20,1,1,1\n".format(frame, frame + 5))
    (sequence / "gt" / "gt.txt").write_text("".join(rows), encoding="utf-8")
    config = replace(
        GSVLTConfig(),
        history_size=3,
        short_term_lost=2,
        max_group_size=3,
        candidate_count=3,
    )
    dataset = GSVLTMOTDataset(sequence, config, seed=1, max_gap=5)
    sample = dataset[0]
    assert sample["past_sequences"].shape == (3, 3, 8)
    assert sample["detections"].shape == (3, 4)
    assert sample["labels"].sum().item() == 1.0
    assert sample["labels"][sample["positive_index"]].item() == 1.0

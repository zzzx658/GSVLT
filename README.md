# GSVLT

This project contains the code for the GSVLT paper. Built on ByteTrack, it implements the GSVLT long-term visual tracking method proposed in the paper, integrating core modules including GVA, GAM, GMVI, and attention to recover long-term lost targets and maintain identity consistency. The project includes complete model, training, inference, and evaluation code as well as MOT17, MOT20, and DanceTrack configurations, and can be directly used for training and multi-object tracking experiments.

## Main File Locations

- **GVA model:** `yolox/tracker/gsvlt/gva.py`
- **GAM module:** `yolox/tracker/gsvlt/gam.py`
- **GMVI model:** `yolox/tracker/gsvlt/gmvi.py`
- **Data processing:** `yolox/tracker/gsvlt/data.py`
- **Configuration file:** `configs/gsvlt/mot17.yaml`

## Run Video Tracking

```bash
python tools/demo_track.py video `
  -f exps/example/mot/yolox_x_mix_det.py `
  -c pretrained/bytetrack_x_mot17.pth.tar `
  --path videos/palace.mp4 `
  --save_result `
  --device gpu `
  --gsvlt `
  --gsvlt-config configs/gsvlt/mot17.yaml `
  --gsvlt-checkpoint checkpoints/gsvlt_mot17.pt

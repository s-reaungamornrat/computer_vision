set current_dirpath=%~dp0
set mmaction_dirpath=D:/dev/mmaction2
cd %mmaction_dirpath%
echo Current directory %cd%

set output_dirpath=D:/results/ucf101
set config_fpath=configs/recognition/x3d/x3d_s_13x6x1_facebook-kinetics400-rgb.py
set checkpoint_fpath=%output_dirpath%/mmaction2-x3d/demo/x3d_s_13x6x1_facebook-kinetics400-rgb_20201027-623825a0.pth
set output_fpath=%output_dirpath%/mmaction2-x3d/demo/output.mp4

python demo/demo.py %config_fpath% %checkpoint_fpath% %output_dirpath%/demo.mp4 data/kinetics/label_map_k400.txt --out-filename %output_fpath% --font-scale 12  --device cpu

cd %current_dirpath%
echo Current directory %cd%

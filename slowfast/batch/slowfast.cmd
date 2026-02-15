set current_dirpath=%~dp0
set mmaction_dirpath=D:/dev/mmaction2
cd %mmaction_dirpath%
echo Current directory %cd%

set output_dirpath=D:/results/ucf101
set config_fpath=configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py
set checkpoint_fpath=%output_dirpath%/mmaction2-slowfast/demo/slowfast_r50_8xb8-4x16x1-256e_kinetics400.pth 
REM slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth
set output_fpath=%output_dirpath%/mmaction2-slowfast/demo/output.mp4

python demo/demo.py %config_fpath% %checkpoint_fpath% %output_dirpath%/demo.mp4 data/kinetics/label_map_k400.txt --out-filename %output_fpath% --font-scale 12  --device cpu

cd %current_dirpath%
echo Current directory %cd%

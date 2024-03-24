DATA_DIR=$1

mkdir "${DATA_DIR}"

echo "Download Acrobot Swingup Task Data"
wget -P ${DATA_DIR} https://obj.umiacs.umd.edu/dmc-pretrain/acrobot_swingup_replay.zip
unzip "${DATA_DIR}/acrobot_swingup_replay.zip" -d "${DATA_DIR}"

echo "Download Finger Turn Hard Task Data"
wget -P ${DATA_DIR} https://obj.umiacs.umd.edu/dmc-pretrain/finger_turn_hard_replay.zip
unzip "${DATA_DIR}/finger_turn_hard_replay.zip" -d "${DATA_DIR}"

echo "Download Hopper Stand Task Data"
wget -P ${DATA_DIR} https://obj.umiacs.umd.edu/dmc-pretrain/hopper_stand_replay.zip
unzip "${DATA_DIR}/hopper_stand_replay.zip" -d "${DATA_DIR}"

echo "Download Walker Run Task Data"
wget -P ${DATA_DIR} https://obj.umiacs.umd.edu/dmc-pretrain/walker_run_replay.zip
unzip "${DATA_DIR}/walker_run_replay.zip" -d "${DATA_DIR}"

echo "Download Humanoid Stand Task Data"
wget -P ${DATA_DIR} https://obj.umiacs.umd.edu/dmc-pretrain/humanoid_stand_replay.zip
unzip "${DATA_DIR}/humanoid_stand_replay.zip" -d "${DATA_DIR}"

echo "Download Dog Walk Task Data"
wget -P ${DATA_DIR} https://obj.umiacs.umd.edu/dmc-pretrain/dog_walk_replay.zip
unzip "${DATA_DIR}/dog_walk_replay.zip" -d "${DATA_DIR}"

echo "Download Evaluation Data"
wget -P ${DATA_DIR} https://obj.umiacs.umd.edu/dmc-pretrain/dmc_eval_data.zip
unzip "${DATA_DIR}/dmc_eval_data.zip" -d "${DATA_DIR}"

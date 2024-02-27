# Premier-TACO is a Few-Shot Policy Learner: Pretraining Multitask Representation via Temporal Action-Driven Contrastive Loss
<p align="center" style="font-size: 50px">
   <a href="https://arxiv.org/abs/2310.19668">[Paper]</a>&emsp;<a href="https://drm-rl.github.io/">[Project Website]</a>
</p>

This branch includes the PyTorch implementation of **Premier-TACO** for **Deepmind Control Suite (DMC)**. (The code for **MetaWorld** and **LIBERO** will be coming soon!) 
Building upon the recent temporal action contrastive learning (TACO) objective, which obtains the state of art performance in visual control tasks, **Premier-TACO** additionally employs a simple yet effective negative example sampling strategy. This strategy is crucial in significantly boosting TACO’s computational efficiency, making large-scale multitask offline pretraining feasible. Our empirical evaluation in a diverse set of continuous control benchmarks including Deepmind Control Suite, MetaWorld, and LIBERO demonstrate Premier-TACO’s effectiveness in pretraining visual representations, significantly enhancing few-shot imitation learning of novel tasks.

# 💾 Download Pretraining Dataset
To download the pretraining and evaluation dataset, run:
```
bash download_dataset.sh ${DATA_DIR} 
```


# 🛠️ Installation Instructions
First, create a virtual environment and install all required packages. 
```bash
conda env create -f conda_env.yml 
conda activate premier-taco-dmc
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

## 💻 Code Usage
To pretrain Premier-TACO representation representation, run
```
python train_premier_taco_dist.py exp_name=EXP_NAME offline_data_dir=${DATA_DIR} 
```
Then after running the pretraining script, by default, the trained encoder will stored under the directory ```exp_local/${EXP_NAME}/encoder.pt```

Then to load the pretrained encoder for downstream few-shot behavior cloning for evaluation, run
```
python train_bc.py task_name=${TASK} seed=${SEED} exp_name=${BC_EXP_NAME} encoder_dir=${ENCODER_CKPT}  offline_data_dir=${DATA_DIR}/dmc_eval_data/${TASK} &
```

Here, ```${ENCODER_CKPT}``` is the directory to the trained encoder checkpoint, ```${offline_data_dir}``` is the directory to the expert demonstration trajectories. The results will be saved under the directory ``exp_local/${BC_EXP_NAME}``.





## 📝 Citation

If you use our method or code in your research, please consider citing the paper as follows:

```
@misc{zheng2024premiertaco,
      title={Premier-TACO is a Few-Shot Policy Learner: Pretraining Multitask Representation via Temporal Action-Driven Contrastive Loss}, 
      author={Ruijie Zheng and Yongyuan Liang and Xiyao Wang and Shuang Ma and Hal Daumé III and Huazhe Xu and John Langford and Praveen Palanisamy and Kalyan Shankar Basu and Furong Huang},
      year={2024},
      eprint={2402.06187},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## 🙏 Acknowledgement
Premier-TACO is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. 

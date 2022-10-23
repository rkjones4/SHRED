We include code/trained models for our baseline methods at [baselines.zip](https://drive.google.com/file/d/1kEJHQsLJCy_XnrQNkj-ZvpDc_oZRtlSy/view?usp=sharing). Put this file in the code/methods directory, and unzip it. 

# Evaluating Baseline Methods

Runs the following commands from the code directory.

Base command to run inference for **FPS**

> python3 make_results.py -mt srd -sn_nr 0 -an_nr 0 -mnr 0 -svn {NAME}

Base command to use **ACD** to generate decomposition

> python3 make_results.py -mt acd -svn {NAME}

Base command to run inference for **PN Seg**. This will involve building cuda kernels for the network. Please see the README in the methods/baselines/pn_seg/ directory for further instructions.

> python3 make_results.py -mt pn_seg -svn {NAME}

Base command to run inference for **WOPL**

> python3 make_results.py -mt wopl -pn_pth methods/baselines/wopl/model_output/wopl/wopl_prior/models/prior_net.pt -mn_pth methods/baselines/wopl/model_output/wopl/wopl_merge/models/merge_net.pt -wm merge -svn {NAME}

Running **L2G** inference is more involved, as it expects a very different PyTorch version (1.01). It requires saving data to disk, running inference code from a separate envrionment on the saved data, writing the predictions to disk, and then loading the predictions from disk.

To save data to l2g_eval_data, run:

> python3 make_results.py -en l2g_eval_data -mt l2g_save

To load predictions that L2G has made from disk, where the predictions are in the l2g_eval_preds directory, run:

> python3 make_results.py -svn {SAVE_NAME} -mt l2g_load -ld l2g_eval_preds

Please see the README in the methods/l2g/ directory for further instructions.

# Retraining Baselines Methods

**PN Seg**

Once again, please see the PartNet specific README in methods/baselines/pn_seg/ to first set up the cuda kernels correctly. Then to train the network, you can run:

> python3 train_method.py -mt pn_seg -en {EXP_NAME}

**WOPL**

WOPL first trains a prior network, with the following command:

> python3 train_method.py -mt wopl -en {EXP_NAME} -wm prior

WOPL then trains a merge network, with the following command:

> python3 train_method.py -mt wopl -en {EXP_NAME} -wm merge -pn_pth methods/baseslines/wopl/model_output/{EXP_NAME}/wopl_prior/models/prior_net.pt

**L2G**

For similiar reasons to evaluation, L2G retraining is quite involved.

For the main paper results, we use the author's released models from https://github.com/tiangeluo/Learning-to-Group .

For further details, please see the README in methods/baselines/l2g/


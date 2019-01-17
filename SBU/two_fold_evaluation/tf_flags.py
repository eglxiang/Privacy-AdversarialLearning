import tensorflow as tf
import datetime
import os
import numpy as np

flags = tf.app.flags

GPU_id = "5,6"
GPU_NUM = int((len(GPU_id)+1)/2)

adversarial_ckpt_file_dir = '../adversarial_training/checkpoint/models/L1Loss_NoLambdaDecay_AvgReplicate_MonitorBudget_MonitorUtility_Resample_18_2.0_0.5'
adversarial_job_name = adversarial_ckpt_file_dir.split('/')[-1]
# All adversarial model ckpts saved:
adversarial_ckpt_file_list = [".".join(f.split(".")[:-1]) for f in os.listdir(adversarial_ckpt_file_dir) if os.path.isfile(os.path.join(adversarial_ckpt_file_dir, f)) and '.data' in f]
# load pretrained adversarial model from this file:
adversarial_ckpt_file = adversarial_ckpt_file_list[np.argmax([adversarial_ckpt_file.split('-')[-1] for adversarial_ckpt_file in adversarial_ckpt_file_list])]
adversarial_ckpt_file = os.path.join(adversarial_ckpt_file_dir, adversarial_ckpt_file)

summary_dir = './summaries/' + adversarial_job_name + '/{}' + datetime.datetime.now().strftime("-%Y%m%d_%H%M%S")
two_fold_eval_ckpt_dir = './checkpoint_eval/' + adversarial_job_name + '/{}'
test_result_dir = './test_result/' + adversarial_job_name + '/{}'

flags.DEFINE_string('adversarial_ckpt_file', adversarial_ckpt_file, 'load pretrained adversarial model from this file')
flags.DEFINE_string('adversarial_job_name', adversarial_job_name, 'such as: AlternativeUpdate-NoRest-M1')
flags.DEFINE_string('summary_dir', summary_dir, 'write summary in this dir')
flags.DEFINE_string('two_fold_eval_ckpt_dir', two_fold_eval_ckpt_dir, 'save budget model ckpt in this dir')
flags.DEFINE_string('test_result_dir', test_result_dir, 'save testing result in this dir')

# GPU section:
flags.DEFINE_string('GPU_id', GPU_id, 'gpu ids')
flags.DEFINE_integer('GPU_NUM', GPU_NUM, 'GPU_NUM')


FLAGS = flags.FLAGS
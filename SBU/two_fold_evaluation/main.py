'''
Two-Fold-Evaluation
First-fold: action (utility) prediction performance is preserved
Second-fold: privacy (budget) prediction performance is suppressed
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, '..')

import errno
import pprint
import time
import yaml
import os
import sys
import re
import datetime
import numpy as np
import tensorflow as tf
from six.moves import xrange
import itertools

from input_data import *
from nets import nets_factory
from degradlNet import residualNet
from loss import *
from utils import *
from functions import placeholder_inputs, create_videos_reading_ops

from tf_flags import FLAGS

sys.path.insert(0, '..')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]='4,5'

# global variables:
continue_from_trained_model = False

cfg = yaml.load(open('params.yml'))
pp = pprint.PrettyPrinter()
# pp.pprint(cfg)

batch_size = cfg['TRAIN']['BATCH_SIZE'] * FLAGS.GPU_NUM
n_batches_eval = int(320/batch_size)

model_max_steps_map = {
    'inception_v1': int(3200*128/batch_size),
    'inception_v2': int(400*128/batch_size),
    'resnet_v1_50': int(400*128/batch_size),
    'resnet_v1_101': int(400*128/batch_size),
    'resnet_v2_50': int(400*128/batch_size),
    'resnet_v2_101': int(400*128/batch_size),
    'mobilenet_v1': int(40000*128/batch_size), # 400 for 4 GPUs 128 batchsize
    'mobilenet_v1_075': int(400*128/batch_size),
    'mobilenet_v1_050': int(1000*128/batch_size), # 1000 for 4 GPUs 128 batchsize
    'mobilenet_v1_025': int(1000*128/batch_size),
}
# Whether we need to train from scratch.
# Among the 10 evaluation models, 8 starts from imagenet pretrained model and 2 starts from scratch
model_train_from_scratch_map = {
    'inception_v1': False,
    'inception_v2': False,
    'resnet_v1_50': False,
    'resnet_v1_101': False,
    'resnet_v2_50': False,
    'resnet_v2_101': False,
    'mobilenet_v1': False,
    'mobilenet_v1_075': False,
    'mobilenet_v1_050': True,
    'mobilenet_v1_025': True,
}
model_name_lst = ['mobilenet_v1', 'mobilenet_v1_075', 'mobilenet_v1_050', 'mobilenet_v1_025',
                    'resnet_v1_50', 'resnet_v1_101', 'resnet_v2_50', 'resnet_v2_101',
                    'inception_v1', 'inception_v2']

# def restore_model(sess, dir, varlist, modulename):
#     regex = re.compile(r'(MobilenetV1_?)(\d*\.?\d*)', re.IGNORECASE)
#     if 'mobilenet' in modulename:
#         varlist = {regex.sub('MobilenetV1', v.name[:-2]): v for v in varlist}
#     if os.path.isfile(dir):
#         saver = tf.train.Saver(varlist)
#         saver.restore(sess, dir)
#         print('#############################Session restored from pretrained model at {}!#############################'.format(dir))
#     else:
#         ckpt = tf.train.get_checkpoint_state(checkpoint_dir=dir)
#         if ckpt and ckpt.model_checkpoint_path:
#             saver = tf.train.Saver(varlist)
#             saver.restore(sess, ckpt.model_checkpoint_path)
#             print('#############################Session restored from pretrained model at {}!#############################'.format(ckpt.model_checkpoint_path))

# build graph:
def build_graph(model_name):
    '''
    Returns:
        graph, init_op, train_op,
        logits_op, acc_op, correct_count_op, loss_op,
        tr_videos_op, tr_actor_labels_op, val_videos_op, val_actor_labels_op, test_videos_op, test_actor_labels_op,
        videos_placeholder, labels_placeholder,
        varlist_budget, varlist_degrad
    '''
    graph = tf.Graph()
    with graph.as_default():
        # global step:
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # placholder inputs for graph:
        videos_placeholder, labels_placeholder, istraining_placeholder = placeholder_inputs(cfg['TRAIN']['BATCH_SIZE'] * FLAGS.GPU_NUM, cfg)
        # degradation models:
        network_fn = nets_factory.get_network_fn(model_name,
                                                num_classes=cfg['DATA']['NUM_CLASSES'],
                                                weight_decay=cfg['TRAIN']['WEIGHT_DECAY'],
                                                is_training=istraining_placeholder)
        # grads, logits, loss list:
        tower_grads = []
        logits_lst = []
        losses_lst = []
        # operation method:
        opt = tf.train.AdamOptimizer(1e-4)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_index in range(0, FLAGS.GPU_NUM):
                with tf.device('/gpu:%d' % gpu_index):
                    print('/gpu:%d' % gpu_index)
                    with tf.name_scope('%s_%d' % ('gpu', gpu_index)) as scope:

                        videos = videos_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]
                        budget_labels = labels_placeholder[gpu_index * cfg['TRAIN']['BATCH_SIZE']:(gpu_index + 1) * cfg['TRAIN']['BATCH_SIZE']]

                        degrad_videos = residualNet(videos, is_video=True)
                        degrad_videos = tf.reshape(degrad_videos, [cfg['TRAIN']['BATCH_SIZE'] * cfg['DATA']['DEPTH'], cfg['DATA']['CROP_HEIGHT'], cfg['DATA']['CROP_WIDTH'], cfg['DATA']['NCHANNEL']])
                        # logits:
                        logits, _ = network_fn(degrad_videos)
                        logits = tf.reshape(logits, [-1, cfg['DATA']['DEPTH'], cfg['DATA']['NUM_CLASSES']])
                        logits = tf.reduce_mean(logits, axis=1, keep_dims=False)
                        # loss:
                        loss = tower_loss_xentropy_sparse(scope, logits, budget_labels)
                        # append list:
                        logits_lst.append(logits)
                        losses_lst.append(loss)

                        # varible list of budget model:
                        varlist_budget = [v for v in tf.trainable_variables() if
                                            any(x in v.name for x in ["InceptionV1", "InceptionV2",
                                            "resnet_v1_50", "resnet_v1_101", "resnet_v2_50", "resnet_v2_101",
                                            'MobilenetV1'])]
                        # varible list of degrade model:
                        varlist_degrad = [v for v in tf.trainable_variables() if v not in varlist_budget]
                        # append grads:
                        tower_grads.append(opt.compute_gradients(loss, varlist_budget))

                        # reuse variables:
                        tf.get_variable_scope().reuse_variables()
        # loss tensor:
        loss_op = tf.reduce_mean(losses_lst)
        # acc tensor:
        logits_op = tf.concat(logits_lst, 0)
        acc_op = accuracy(logits_op, labels_placeholder)
        # how many is correctly classified:
        correct_count_op = tf.reduce_sum(
                tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits_op), axis=1), labels_placeholder), tf.int32))
        # grads tensor:
        grads = average_gradients(tower_grads) # average gradient over all GPUs

        # apply gradients operation:
        with tf.control_dependencies([tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))]):
            train_op = opt.apply_gradients(grads, global_step=global_step)

        # input operations:
        tr_videos_op, _, tr_actor_labels_op = create_videos_reading_ops(is_train=True, is_val=False, cfg=cfg)
        val_videos_op, _, val_actor_labels_op = create_videos_reading_ops(is_train=False, is_val=True, cfg=cfg)
        test_videos_op, _, test_actor_labels_op = create_videos_reading_ops(is_train=False, is_val=False, cfg=cfg)
        # initialize operations:
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

        return (graph, init_op, train_op,
                logits_op, acc_op, correct_count_op, loss_op,
                tr_videos_op, tr_actor_labels_op, val_videos_op, val_actor_labels_op, test_videos_op, test_actor_labels_op,
                videos_placeholder, labels_placeholder, istraining_placeholder,
                varlist_budget, varlist_degrad)

def run_training(model_name, pretrained_budget_model_ckpt_dir):
    '''
    Args:
        model_name: name of the testing budget model.
        pretrained_budget_model_ckpt_dir: where to load the pretrained n budget models.
    '''
    # Save ckpt of two-fold eval process in this directory:
    two_fold_eval_ckpt_dir = FLAGS.two_fold_eval_ckpt_dir.format(model_name)
    if not os.path.exists(two_fold_eval_ckpt_dir):
        os.makedirs(two_fold_eval_ckpt_dir)
    # Save summary files in this dir:
    summary_dir = FLAGS.summary_dir.format(model_name)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    train_summary_file = open(summary_dir + '/train_summary.txt', 'w')
    val_summary_file = open(summary_dir + '/val_summary.txt', 'w')

    # build graph:
    (graph, init_op, train_op,
    _, acc_op, correct_count_op, loss_op,
    tr_videos_op, tr_actor_labels_op, val_videos_op, val_actor_labels_op, _, _,
    videos_placeholder, labels_placeholder, istraining_placeholder,
    varlist_budget, varlist_degrad) = build_graph(model_name)

    # session configuration:
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    # run session:
    with tf.Session(graph=graph, config=config) as sess:
        '''
        In training, first run init_op, then do multi-threads.
        '''
        # initialize variables:
        sess.run(init_op)

        # multi threads:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        # Load ckpts:
        bn_moving_vars = [g for g in tf.global_variables() if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in tf.global_variables() if 'moving_variance' in g.name]

        if continue_from_trained_model:
            varlist = tf.trainable_variables()
            varlist += bn_moving_vars
            saver = tf.train.Saver(varlist)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir=two_fold_eval_ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('#############################Session restored from trained model at {}!###############################'.format(ckpt.model_checkpoint_path))
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), two_fold_eval_ckpt_dir)
        else:
            if not model_train_from_scratch_map[model_name]:
                saver = tf.train.Saver(varlist_degrad)
                saver.restore(sess, FLAGS.adversarial_ckpt_file)
                #saver.restore(sess, '../adversarial_training/checkpoint/models/L1Loss_NoLambdaDecay_AvgReplicate_MonitorBudget_MonitorUtility_Resample_18_2.0_0.5')

                varlist = [v for v in varlist_budget+bn_moving_vars if not any(x in v.name for x in ["logits"])]
                restore_model_ckpt(sess=sess, ckpt_dir=pretrained_budget_model_ckpt_dir, varlist=varlist, modulename=model_name)
        # End loading ckpts.

        # saver for saving all trainable variables (budget model+degrade model) ckpts:
        saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars, max_to_keep=5)

        best_val_acc = 0
        best_acc_step = -1
        for step in xrange(model_max_steps_map[model_name]):
            # updata on training data:
            start_time = time.time()
            train_videos, train_labels = sess.run([tr_videos_op, tr_actor_labels_op])
            _, acc, loss_value = sess.run([train_op, acc_op, loss_op],
                feed_dict={videos_placeholder: train_videos, labels_placeholder: train_labels, istraining_placeholder: True})
            assert not np.isnan(np.mean(loss_value)), 'Model diverged with loss = NaN'
            #print(sess.run(bn_moving_vars[0]))
            # print summary:
            if step % cfg['TRAIN']['PRINT_STEP'] == 0:
                summary = 'Step: {:4d}, time: {:.4f}, accuracy: {:.5f}, loss: {:.8f}'.format(step, time.time() - start_time, acc, np.mean(loss_value))
                print(summary)
                train_summary_file.write(summary + '\n')

            # validation on val set and save ckpt:
            if step % cfg['TRAIN']['VAL_STEP'] == 0 or (step + 1) == model_max_steps_map[model_name]:
                start_time = time.time()

                val_acc_lst = []
                for _ in itertools.repeat(None, n_batches_eval):
                    val_videos, val_labels = sess.run([val_videos_op, val_actor_labels_op])
                    [val_acc] = sess.run([acc_op],
                        feed_dict={videos_placeholder: val_videos, labels_placeholder: val_labels, istraining_placeholder: False})
                    val_acc_lst.append(val_acc)

                acc = np.mean(val_acc_lst)
                # end calculating val_acc
                # start summary:
                summary = ("Step: {:4d}, time: {:.4f}, validation accuracy: {:.5f}").format(
                                    step, time.time() - start_time, acc)
                print('Validation:\n' + summary)
                val_summary_file.write(summary + '\n')
                # end summary
                # start saving model
                if acc > best_val_acc:
                    # update best_val_acc:
                    best_val_acc = acc
                    best_acc_step = step
                    print('Get new best val_acc: %f\n' % best_val_acc)
                # Save checkpoint:
                checkpoint_path = os.path.join(two_fold_eval_ckpt_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                # end saving model

        # join multi threads:
        coord.request_stop()
        coord.join(threads)

    print("done")


def run_testing(model_name):
    # save testing result in this dir:
    test_result_dir = FLAGS.test_result_dir.format(model_name)
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)
    test_result_file = open(test_result_dir + '/test_result' + datetime.datetime.now().strftime("-%Y%m%d_%H%M%S") + '.txt', 'w')
    test_log_file = open(test_result_dir + '/test_log'+ datetime.datetime.now().strftime("-%Y%m%d_%H%M%S") + '.txt', 'w+')

    # build graph:
    (graph, init_op, _,
    logits_op, _, correct_count_op, _,
    tr_videos_op, tr_actor_labels_op, val_videos_op, val_actor_labels_op, test_videos_op, test_actor_labels_op,
    videos_placeholder, labels_placeholder, istraining_placeholder,
    varlist_budget, varlist_degrad) = build_graph(model_name)

    # session config:
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    # run session:
    with tf.Session(graph=graph, config=config) as sess:
        # run initialization:
        sess.run(init_op)
        # multi threads:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # load degrade and budget model ckpts:
        bn_moving_vars = [g for g in tf.global_variables() if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in tf.global_variables() if 'moving_variance' in g.name]
        saver = tf.train.Saver(tf.trainable_variables() + bn_moving_vars)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.two_fold_eval_ckpt_dir.format(model_name))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Session restored from pretrained budget model at {}!'.format(ckpt.model_checkpoint_path))
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), FLAGS.two_fold_eval_ckpt_dir.format(model_name))
        # end loading ckpts

        total_v = 0.0
        test_correct_num = 0.0
        c = 0
        try:
            while not coord.should_stop():
                c += 1
                print('in while loop ', str(c))
                videos, labels = sess.run([test_videos_op, test_actor_labels_op])
                total_v += labels.shape[0]
                if videos.shape[0] < batch_size: # the last batch of testing data
                    videos = np.pad(videos, ((0,batch_size-videos.shape[0]),(0,0),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
                    labels = np.pad(labels, ((0,batch_size-labels.shape[0])), 'constant', constant_values=-1)
                    # the padded videos will never be true, since it can never be classified as -1
                print('videos:', videos.shape)
                print('labels:', labels.shape)
                [correct_num, logtis ] = sess.run([correct_count_op, logits_op],
                    feed_dict={videos_placeholder: videos, labels_placeholder: labels, istraining_placeholder: True})
                print('correct_num:', correct_num)
                assert correct_num <= videos.shape[0]
                test_correct_num += correct_num
                # test_log_file.write('logits: {}\nlabels: {}\nvideos: {}'.format(logtis[0,0:5], labels, videos[0,0,0:9,0,0]))
        except tf.errors.OutOfRangeError:
            print('Done testing on all the examples')
        finally:
            coord.request_stop()

        # print and write testing result:
        test_result_str = 'test acc: {}\ttest_correct_num:{}\ttotal_v:{}\n'.format(test_correct_num / total_v, test_correct_num, total_v)
        print(test_result_str)
        test_result_file.write(test_result_str)
        # print('logits:', logtis)
        # print('labels:', labels)
        # print('videos:', videos[0,0,0:9,0,0])


        # close multiple threads:
        coord.join(threads)
        sess.close()

    print("done")

def main():
    # evaluate using N different budget models:
    for model_name in model_name_lst[0:1]:
        # load pretrained eval model initialization from this dir:
        pretrained_budget_model_ckpt_dir = '../evaluation_models/' + 'mobilenet_v1_1.0_128'

        # training:
        #run_training(model_name = model_name, pretrained_budget_model_ckpt_dir = pretrained_budget_model_ckpt_dir)
        # testing:
        run_testing(model_name = model_name)

if __name__ == '__main__':
    main()
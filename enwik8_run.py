from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from copy import deepcopy
import time
import reader
import os
import rnn_cell_additions as dr

SESS_CONFIG = tf.ConfigProto()
SESS_CONFIG.gpu_options.allow_growth = True

NAME = 'rhn'

INIT_SCALE = None
LEARNING_RATE = None
MAX_GRAD_NORM = None

NUM_LAYERS = None
UNITS_NUM = 1500
TIME_STEPS = None
DROP_OUTPUT_INIT = None
DROP_OUTPUT_STEP = None
DROP_STATE_INIT = None
DROP_STATE_STEP = None
DROP_EMB = None
TWOS_WEIGHT = None
OUT_SIZE = None
BATCH_SIZE = None
TRAIN_SIZE = None
VALID_SIZE = None
TEST_SIZE = None
FORGET_BIAS_INIT = None
SWEEPS = None
RESTORE_PATH = None
RESTORE = None

TWOSTHRESH = None

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("name", NAME, "Simulation name")
flags.DEFINE_string("restore_path", RESTORE_PATH, "restore path")

flags.DEFINE_integer("num_layers", NUM_LAYERS, "NUM_LAYERS")
flags.DEFINE_integer("max_grad_norm", MAX_GRAD_NORM, "max_grad_norm")
flags.DEFINE_integer("units_num", UNITS_NUM, "units_num")
flags.DEFINE_integer("time_steps", TIME_STEPS, "time_steps")
flags.DEFINE_integer("sweeps", SWEEPS, "sweeps")
flags.DEFINE_integer("batch_size", BATCH_SIZE, "BATCH_SIZE")
flags.DEFINE_integer("train_size", TRAIN_SIZE, "train_size")
flags.DEFINE_integer("valid_size", VALID_SIZE, "valid_size")
flags.DEFINE_integer("test_size", TEST_SIZE, "test_size")

flags.DEFINE_float("init_scale", INIT_SCALE, "init_scale")
flags.DEFINE_float("learning_rate", LEARNING_RATE, "learning_rate")
flags.DEFINE_float("drop_output_init", DROP_OUTPUT_INIT, "drop_output_init")
flags.DEFINE_float("drop_output_step", DROP_OUTPUT_STEP, "drop_output_step")
flags.DEFINE_float("drop_state_init", DROP_STATE_INIT, "drop_state_init")
flags.DEFINE_float("drop_state_step", DROP_STATE_STEP, "drop_state_step")
flags.DEFINE_float("drop_emb", DROP_EMB, "drop_emb")
flags.DEFINE_float("forget_bias_init", FORGET_BIAS_INIT, "forget_bias_init")

flags.DEFINE_boolean("restore", RESTORE, "restore option")

# ex = Experiment('rhn_prediction')
logging = tf.logging


class Config(object):
    """ config."""
    name = 'trail'
    dataset = 'enwik8'
    restore = True
    restore_path = '../Saved_Models/1_layer_trained/saver/model.ckpt-1079940'
    init_scale = 0.05
    learning_rate = 0.5
    max_grad_norm = 5
    time_steps = 10
    units_num = 1500
    lr_sat_mark = 0.01
    max_max_epoch = 80
    drop_emb = 0.2
    keep_prob_state = 1.0 - drop_emb
    lr_decay = 0.8
    batch_size = 100
    vocab_size = 205 if dataset == 'enwik8' else 10000
    forget_bias_init = 0.0
    num_layers = 2
    init_epoch = 25
    sweeps = 0
    heuristic_num = 1
    variational = True
    drop_output_init = 0.1
    drop_output_step = 0.05
    drop_state_init = 0.1
    drop_state_step = 0.05
    drop_output = [[0.1, 0.1, 0.0, 0.0], [0.1, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    drop_state = [[0.1, 0.1, 0.0, 0.0], [0.1, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    train_size = 1000000
    valid_size = 100000
    test_size = 5000000

def get_config():
  C = Config()
  config = read_flags(C, FLAGS)
  return config

def read_flags(config, FLAGS):
  # assign flags into config
  flags_dict = FLAGS.__dict__['__flags']
  for key, val in flags_dict.items():
      if val is not None:
          setattr(config, key, val)
  train_size_mil = config.train_size / 1000000
  simulation_name = 'layers' + str(config.num_layers) \
                     + '_hidden' + str(config.units_num) \
                     + '_batch' + str(config.batch_size) \
                     + '_numsteps' + str(config.time_steps) + '_' \
                     + str(int(train_size_mil)) + 'Mexamples'\
                     + '_GL' + str(int(config.sweeps)) \
                     + '_LAD' + str(config.heuristic_num) \
                     + '_lr_sat' + str(config.lr_sat_mark) \
                     + '_DOemb' + str(config.drop_emb) \
                     + '_DOout' + str(config.drop_output_init) + '-' + str(config.drop_output_step)\
                     + '_DOstate' + str(config.drop_state_init) + '-' + str(config.drop_state_step)
  for i in range(config.num_layers):
      for j in range(config.num_layers):
          config.drop_state[i][j] = config.drop_state_init + j*config.drop_state_step
          config.drop_output[i][j] = config.drop_output_init + j*config.drop_output_step
  config.sim_name = config.name + '/' + simulation_name
  return config

config = get_config()

class PTBModel(object):
    """class for handling the learning of the ptb model"""

    # batch_size,time_steps,vocab_size,units_num,drop_emb,forget_bias_init,num_layers,max_grad_norm
    def __init__(self,
                 config,
                 is_training,
                 inputs):
        """ the constructor defines how the data flow of the model happens"""
        self.config = config
        self._input = inputs
        batch_size = inputs.batch_size  # num of sequences
        time_steps = config.time_steps  # num of time steps every grad
        vocab_size = config.vocab_size  # num of possible words
        units_num = config.units_num  # num of elements in the hidden layer
        self.drop_emb = config.drop_emb
        ############# embedding layer #############
        with tf.variable_scope("embedding"):
            with tf.device("/cpu:0"):
                embedding_map = tf.get_variable(
                    name="embedding", shape=[vocab_size, units_num], dtype=tf.float32)
                b_embed_in = tf.get_variable(name="b_embed_in", shape=[units_num], dtype=tf.float32)
                embedding = tf.nn.embedding_lookup(embedding_map, self._input.input_data) + b_embed_in

            if is_training and config.keep_prob_state < 1:
                embedding_out = tf.nn.dropout(embedding, (1.0-config.drop_emb))
            else:
                embedding_out = embedding

        ############# lstm layers #############

        # define basic lstm cell
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(
                num_units=units_num, forget_bias=config.forget_bias_init, state_is_tuple=True)

        possible_cell = lstm_cell
        # if dropout is needed add a dropout wrapper
        if is_training and config.drop_emb > 0:
            def possible_cell():
                if config.variational:
                    return dr.VariationalDropoutWrapperModified(lstm_cell(), batch_size, units_num,
                                                                output_keep_prob=(1.0-config.drop_emb),
                                                                state_keep_prob=config.keep_prob_state)
                else:
                    return dr.DropoutWrapper(
                        lstm_cell(), output_keep_prob=(1.0-config.drop_emb))


        # feed forward through lstm layers

        # build the lstm layers accordingly
        self.cell = []
        self._initial_state = []
        for i in range(config.num_layers):
            self.cell.append(possible_cell())
            self._initial_state.append(self.cell[i].zero_state(batch_size, dtype=tf.float32))

        # initiate outputs and states
        outputs = []
        state = []
        lstm_output = []
        for i in range(config.num_layers):
            outputs.append([])
            state.append([])

        with tf.variable_scope("lstm_layer_" + str(1)):
            state[0] = self._initial_state[0]

            for time_step in range(time_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (new_h, state[0]) = self.cell[0](embedding_out[:, time_step, :], state[0])
                outputs[0].append(new_h)
            lstm_output.append(tf.reshape(tf.concat(values=outputs[0], axis=1), [-1, units_num]))

        for i in range(1, config.num_layers):
            with tf.variable_scope("lstm_layer_" + str(i + 1)):
                state[i] = self._initial_state[i]

                for time_step in range(time_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (new_h, state[i]) = self.cell[i](outputs[i - 1][time_step], state[i])
                    outputs[i].append(new_h)

                lstm_output.append(tf.reshape(tf.concat(values=outputs[i], axis=1), [-1, units_num]))

        w = []
        b_embed_out = tf.get_variable(name="b_embed_out", shape=[vocab_size], dtype=tf.float32)
        logits = []

        # loss calculation
        loss = []
        self._cost = []
        for i in range(config.num_layers):
            # softmax layer
            with tf.variable_scope("loss" + str(i + 1)):
                w.append(tf.transpose(embedding_map))
                # w.append(tf.get_variable(name="w_softmax", shape=[units_num, vocab_size], dtype=tf.float32))
                # b.append(tf.get_variable(name="b_softmax", shape=[vocab_size], dtype=tf.float32))

                logits.append(tf.matmul(lstm_output[i], w[i]) + b_embed_out)
                loss.append(tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                    [logits[i]], [tf.reshape(inputs.targets, [-1])],
                    [tf.ones([batch_size * time_steps], dtype=tf.float32)]))
                self._cost.append(tf.reduce_sum(loss[i]) / batch_size)
        cost = self._cost
        self._final_state = state

        # define training procedure for training data
        if not is_training:
            return

        # set learning rate as variable in order to anneal it throughout training
        self._lr = tf.Variable(0.0, trainable=False)

        # get trainable vars
        tvars = tf.trainable_variables()
        grads = []
        optimizer = []
        self._train_op = []
        for i in range(config.num_layers):
            with tf.name_scope("optimizer" + str(i + 1)):
                # apply grad clipping
                grad, _ = tf.clip_by_global_norm(tf.gradients(cost[i], tvars), config.max_grad_norm)
                grads.append(grad)
                # define optimizer
                optimizer.append(tf.train.GradientDescentOptimizer(self._lr))
                # define the train operation with the normalized grad
                self._train_op.append(optimizer[-1].apply_gradients(
                    zip(grads[-1], tvars),
                    global_step=tf.contrib.framework.get_or_create_global_step()))
        with tf.name_scope("learning_rate"):
            # a placeholder to assign a new learning rate
            self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")

            # function to update learning rate
            self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def update_masks(self, session):
        for i in range(config.num_layers):
            self.cell[i].update_masks(session)

    def update_drop_params(self, session, output_drop, state_drop):
        for i in range(config.num_layers):
            output_keep_prob = 1.0 - output_drop[i]
            state_keep_prob = 1.0 - state_drop[i]
            self.cell[i].update_drop_params(session,
                                            output_keep_prob,
                                            state_keep_prob)

    def print_tvars(self):
        tvars = tf.trainable_variables()
        nvars = 0
        for var in tvars[1:]:
          sh = var.get_shape().as_list()
          nvars += np.prod(sh)
          print('var: %s, size: [%s]' % (var.name,', '.join(map(str, sh))))
        print(nvars, 'total variables')

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    def cost(self, layer):
        return self._cost[layer]

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    def train_op(self, layer=-1):
        return self._train_op[layer]


class PTBInput(object):
    """The input data."""

    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.time_steps = time_steps = config.time_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // time_steps
        self.input_data, self.targets = reader.data_producer(
            data, batch_size, time_steps, name=name)


def run_epoch(session, model, eval_op=None, verbose=True, layer=-1):
    """run the given model over its data"""

    start_time = time.time()
    costs = 0.0
    iters = 0

    # zeros initial state
    state = session.run(model.initial_state)
    if config.variational and eval_op is not None:
        model.update_masks(session)
    # determine the evaluations that are done every epoch
    fetches = {
        "cost": model.cost(layer),
        "final_state": model.final_state,
    }
    # if training is needed
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):

        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.input.time_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f bits: %.3f speed: %.0f cps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iters), np.log2(np.exp(costs / iters)),
                   iters * model.input.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)

def main(_):
    epoch_bias = 0
    data_path = 'data'
    config = get_config()
    eval_config = get_config()
    # eval_config.batch_size = 1
    eval_config.time_steps = eval_config.time_steps
    dataset = config.dataset
    # if dataset == 'ptb':
    #     data_path = "./PTB_data"
    #     raw_data = reader.ptb_raw_data(data_path)
    if dataset == 'enwik8':
        from data import reader
        raw_data = reader.enwik8_raw_data(data_path, num_test=config.test_size, num_train=config.train_size)

    train_data, valid_data, test_data, _ = raw_data
    simulation_name = config.sim_name
    directory = "../logs/" + simulation_name
    i = 1
    if os.path.exists(directory):
        directory = directory + '_' + str(i)
    while os.path.exists(directory):
        i += 1
        directory = directory[:-1] + str(i)

    if not os.path.exists(directory):
        print('Save directory in %s' % directory)
        os.makedirs(directory)
        os.makedirs(directory + '/train')
        os.makedirs(directory + '/test')
        os.makedirs(directory + '/saver')

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")

        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config, inputs=train_input)

        train_writer = tf.summary.FileWriter(directory + '/train',
                                             graph=tf.get_default_graph())
        tf.summary.scalar("Training Loss", m.cost(layer=-1))
        tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config, inputs=valid_input)
        tf.summary.scalar("Validation Loss", mvalid.cost(layer=-1))

        valid_writer = tf.summary.FileWriter(directory + '/valid')

    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=False, config=eval_config, inputs=test_input)
        tf.summary.scalar("Test Loss", mtest.cost(layer=-1))
        test_writer = tf.summary.FileWriter(directory + '/test')
    initer = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # create trainable variables saver to restore previous trained layers and train save the new
    tvars_res = tf.trainable_variables()
    if config.restore and config.num_layers > 1:
        rmv_idx = 2*config.num_layers
        del tvars_res[rmv_idx]
        del tvars_res[rmv_idx]
    trainable_saver = tf.train.Saver(var_list=tvars_res)

    with tf.name_scope("summaries"):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sv = tf.train.Supervisor(logdir=directory + '/saver')
        with sv.managed_session(config=sess_config) as session:
            current_lr = None

            session.run(initer)
            if config.restore:
                print('restoring model from: %s' % config.restore_path)
                trainable_saver.restore(session, config.restore_path)
                print('restore completed')
                m.print_tvars()

            #### Restoring:
            # save_path = 'D:\Machine_learning\PTB\gradualTrain\gradualTrain_4layers_hidden1125_steps50_GL_LAD\layer3_model.ckpt-2615532'
            # saver.restore(session, save_path)
            # m.print_tvars()
            # ####

            m.update_drop_params(session, config.drop_output[-1], config.drop_state[-1])
            for sweep in range(config.sweeps):
                print("\n\nsweep #%d" % (sweep+1))
                start_time = time.time()
                for layer in range(0, config.num_layers):
                    print("training layer #%d" % (layer + 1))
                    valid_perplexity = []

                    m.update_drop_params(session, config.drop_output[layer], config.drop_state[layer])
                    heuristic_decay_counter = 1
                    # for i in range(0, config.init_epoch):
                    i = -1
                    current_lr = config.learning_rate
                    while current_lr > config.lr_sat_mark:
                        i += 1
                        if i > 1:
                            if valid_perplexity[-1] > valid_perplexity[-2]:
                                if heuristic_decay_counter == config.heuristic_num:
                                    current_lr = session.run(m.lr)
                                    current_lr *= config.lr_decay
                                    m.assign_lr(session, current_lr)
                                    heuristic_decay_counter = 1
                                else:
                                    heuristic_decay_counter += 1
                        elif i == 0:
                            current_lr = config.learning_rate
                            m.assign_lr(session, current_lr)

                        print("\nEpoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
                        train_perplexity = run_epoch(session, m, eval_op=m.train_op(layer), verbose=True, layer=layer)
                        train_sum = tf.Summary(
                            value=[tf.Summary.Value(tag="train_perplexity_s" + str(sweep + 1) + "l" + str(layer),
                                                    simple_value=train_perplexity)])
                        train_writer.add_summary(train_sum, i + 1 + epoch_bias)
                        print("Epoch: %d Train Perplexity: %.3f bits: %.3f" % (i + 1, train_perplexity, np.log2(train_perplexity)))
                        valid_perplexity.append(run_epoch(session, mvalid, verbose=False, layer=layer))
                        valid_sum = tf.Summary(
                            value=[tf.Summary.Value(tag="valid_perplexity_s" + str(sweep + 1) + "l" + str(layer),
                                                    simple_value=valid_perplexity[i])])
                        valid_writer.add_summary(valid_sum, i + 1 + epoch_bias)
                        print("Epoch: %d Valid Perplexity: %.3f bits: %.3f" % (i + 1, valid_perplexity[i], np.log2(valid_perplexity[i])))

                        # test_perplexity = run_epoch(session, mtest, verbose=False, layer=layer)
                        # test_sum = tf.Summary(
                        #     value=[tf.Summary.Value(tag="test_perplexity_s" + str(sweep + 1) + "l" + str(layer),
                        #                             simple_value=test_perplexity)])
                        # test_writer.add_summary(test_sum, i + 1 + epoch_bias)
                        # print("Epoch: %d Test Perplexity: %.3f bits: %.3f" % (i + 1, test_perplexity, np.log2(test_perplexity)))
                    if not os.path.exists(directory + '/layer' + str(layer + 1) + '_model'):
                        os.makedirs(directory + '/layer' + str(layer + 1) + '_model')
                    save_path = sv.saver.save(session, directory + '/layer'
                                              + str(layer+1) + '_model.ckpt',
                                              global_step=sv.global_step)
                    print("Saving layer to %s." % save_path)
                    test_perplexity = run_epoch(session, mtest, verbose=False, layer=layer)
                    test_sum = tf.Summary(
                        value=[tf.Summary.Value(tag="test_perplexity_s" + str(sweep + 1) + "l" + str(layer),
                                                simple_value=test_perplexity)])
                    test_writer.add_summary(test_sum, i + 1 + epoch_bias)
                    print("Epoch: %d Test Perplexity: %.3f bits: %.3f" % (1, test_perplexity, np.log2(test_perplexity)))
                elapsed = time.time() - start_time
                print("initialization took %02d:%02d:%02d\n" % (elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))

            if current_lr is None:
                current_lr = config.learning_rate
            start_time = time.time()
            valid_perplexity = []
            heuristic_decay_counter = 1
            for i in range(config.max_max_epoch):
                if i > 1:
                    if valid_perplexity[-1] > valid_perplexity[-2]:
                        if heuristic_decay_counter == config.heuristic_num:
                            current_lr *= config.lr_decay
                            heuristic_decay_counter = 1
                        else:
                            heuristic_decay_counter += 1

                if current_lr < config.lr_sat_mark:
                    print('learning rate under saturation level - finished')
                    break
                m.assign_lr(session, current_lr)

                print("\nEpoch: %d Learning rate: %.6f" % (i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op(-1), verbose=True)
                train_sum = tf.Summary(
                    value=[tf.Summary.Value(tag="train_perplexity_end", simple_value=train_perplexity)])
                train_writer.add_summary(train_sum, i + 1 + epoch_bias)
                print("Epoch: %d Train Perplexity: %.3f bits: %.3f" % (i + 1, train_perplexity, np.log2(train_perplexity)))
                valid_perplexity.append(run_epoch(session, mvalid, verbose=False))
                valid_sum = tf.Summary(
                    value=[tf.Summary.Value(tag="valid_perplexity_end", simple_value=valid_perplexity[i])])
                valid_writer.add_summary(valid_sum, i + 1 + epoch_bias)
                print("Epoch: %d Valid Perplexity: %.3f bits: %.3f" % (i + 1, valid_perplexity[i], np.log2(valid_perplexity[i])))

                # test_perplexity = run_epoch(session, mtest, verbose=False)
                # test_sum = tf.Summary(
                #     value=[tf.Summary.Value(tag="test_perplexity_end", simple_value=test_perplexity)])
                # test_writer.add_summary(test_sum, i + 1 + epoch_bias)
                # print("Epoch: %d Test Perplexity: %.3f bits: %.3f" % (i + 1, test_perplexity, np.log2(test_perplexity)))

            end_time = time.time()
            elapsed = end_time - start_time
            print("optimization took %02d:%02d:%02d\n" % (elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))

            test_perplexity = run_epoch(session, mtest, verbose=False)
            test_sum = tf.Summary(
                value=[tf.Summary.Value(tag="test_perplexity_end", simple_value=test_perplexity)])
            test_writer.add_summary(test_sum, i + 1 + epoch_bias)
            print("Epoch: %d Test Perplexity: %.3f bits: %.3f" % (i + 1, test_perplexity, np.log2(test_perplexity)))

            if True:
                print("Saving model to %s." % (directory + '/saved_model'))
                sv.saver.save(session, directory + '/saved_model')
                trainable_saver.save(session, directory + '/saved_model_tvars')

if __name__ == "__main__":
    tf.app.run()

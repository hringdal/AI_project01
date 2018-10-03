import tensorflow as tf
import numpy as np
import matplotlib.pyplot as PLT
import tflowtools2 as TFT
import os
import json


class Gann:

    def __init__(self, dims, cman, settings, learning_rate=.1, mbs=10, vint=None, softmax=False):
        self.learning_rate = learning_rate
        self.layer_sizes = dims  # Sizes of each layer of neurons
        self.global_training_step = 0  # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.caseman = cman
        self.softmax_outputs = softmax
        self.modules = []
        self.settings = settings
        self.build()

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type,spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self,module_index,type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def get_values(self, modules, type='out'):

        length_cases = len(self.caseman.get_testing_cases())
        indices = np.random.choice(length_cases, self.settings['map_batch_size'])
        samples = self.caseman.get_testing_cases()[indices]
        for module in modules:
            self.add_grabvar(module, type=type)

        inputs = [c[0] for c in samples]
        targets = [c[1] for c in samples]

        feeder = {self.input: inputs, self.target: targets}

        _, values, session = self.run_one_step(self.predictor, self.grabvars, session=self.current_session, feed_dict=feeder)

        self.grabvars = []
        return values, targets

    def do_mapping(self, modules):
        map_vals, _ = self.get_values(modules)
        for layer in range(len(modules)):
            TFT.hinton_plot(map_vals[layer], title='mapping layer ' + str(modules[layer]))

    def do_dendrogram(self, modules):
        values, targets = self.get_values(modules)
        for layer in range(len(modules)):
            TFT.dendrogram(values[layer], [TFT.bits_to_str(bits) for bits in targets], title="dendrogram layer"+str(modules[layer]))

    def do_wgt_bias_view(self, modules, type):
        values, _ = self.get_values(modules, type=type)

        for layer in range(len(modules)):
            print(layer)
            layer_value = values[layer]
            if type == 'bias':
                layer_value = np.reshape(values[layer], (1, -1))

            TFT.display_matrix(layer_value, title=str(type) + ' in layer ' + str(modules[layer]))

    def add_module(self,module):
        self.modules.append(module)

    def build(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input; insize = num_inputs
        # Build all of the modules
        for i, outsize in enumerate(self.layer_sizes[1:]):

            if i == len(self.layer_sizes) - 1:
                gmod = Gannmodule(self, i, invar, insize, outsize, is_output=True)
            else:
                gmod = Gannmodule(self,i,invar,insize,outsize)

            invar = gmod.output; insize = gmod.outsize

        self.output = gmod.output  # Output of last module is output of whole network

        if self.softmax_outputs:
            self.output = tf.nn.softmax(self.output)

        self.target = tf.placeholder(tf.float64,shape=(None,gmod.outsize),name='Target')
        self.configure_learning()

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self):
        # configure cost function
        cost_func = self.settings['cost_function']

        if cost_func == 'rmse':
            self.error = tf.sqrt(tf.reduce_mean(tf.square(self.target - self.output)),  name='RMSE')
        elif cost_func == 'cross-entropy':
            self.error = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target, logits=self.output)
        else:
            # defaults to MSE
            self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')

        self.predictor = self.output  # Simple prediction runs will request the value of output neurons

        # configure optimizer
        opt_name = self.settings['optimizer']

        if opt_name == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif opt_name == 'gradient':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif opt_name == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif opt_name == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        else:
            # defaults to gradient descent
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.trainer = optimizer.minimize(self.error, name='Backprop')

    def do_training(self, sess, cases, steps=100, continued=False):
        if not(continued):
            self.error_history = []

        '''
        for i in range(epochs):
            error = 0; step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size; ncases = len(cases); nmb = math.ceil(ncases/mbs)

            
            for cstart in range(0,ncases,mbs):  # Loop through cases, one minibatch at a time.
                cend = min(ncases,cstart+mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]; targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _,grabvals,_ = self.run_one_step([self.trainer],gvars,self.probes,session=sess,
                                         feed_dict=feeder,step=step)
                error += grabvals[0]

            self.error_history.append((step, error/nmb))
            self.consider_validation_testing(step,sess)
            '''

        for i in range(steps):
            error = 0
            step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size
            ncases = len(cases)

            cases = np.array(cases)
            idx = np.random.choice(ncases, mbs, replace=False)
            minibatch = cases[idx]

            inputs = [c[0] for c in minibatch]
            targets = [c[1] for c in minibatch]

            feeder = {self.input: inputs, self.target: targets}
            _, grabvals, _ = self.run_one_step([self.trainer], gvars, self.probes, session=sess,
                                               feed_dict=feeder, step=step)

            error += grabvals[0]

            if i % self.validation_interval == 0:
                print('Training Set Error = {}'.format(grabvals[0]))

            self.error_history.append((step, error))
            self.consider_validation_testing(step,sess)

        self.global_training_step += steps

        TFT.plot_training_history(self.error_history,self.validation_history,xtitle="Epoch",ytitle="Error",
                                  title="",fig=not(continued))

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard MSE error function is used for testing.

    def do_testing(self,sess,cases,msg='Testing',bestk=None):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]

        feeder = {self.input: inputs, self.target: targets}

        self.test_func = self.error

        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor,[TFT.one_hot_to_int(list(v)) for v in targets],k=bestk)

        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                           feed_dict=feeder)
        if bestk is None or bestk == 0:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(testres/len(cases))))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns an OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k) # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self,epochs,sess=None,dir="probeview",continued=False):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.roundup_probes() # this call must come AFTER the session is created, else graph is not in tensorboard.
        self.do_training(session,self.caseman.get_training_cases(),epochs,continued=continued)

    def testing_session(self,sess,bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess,cases,msg='Final Testing',bestk=bestk)

    def consider_validation_testing(self,epoch,sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess,cases,msg='Validation')
                self.validation_history.append((epoch,error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self,sess,bestk=None):
        self.do_testing(sess,self.caseman.get_training_cases(),msg='Total Training',bestk=bestk)

    # Similar to the "quickrun" functions used earlier.

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                     session=None, feed_dict=None, step=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)

        return results[0], results[1], sess

    def run(self,epochs=100,sess=None,continued=False,bestk=None):
        self.training_session(epochs,sess=sess,continued=continued)
        self.test_on_trains(sess=self.current_session,bestk=bestk)
        self.testing_session(sess=self.current_session,bestk=bestk)
        self.close_current_session(view=False)

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self,epochs=100,bestk=None):
        self.reopen_current_session()
        self.run(epochs,sess=self.current_session,continued=True,bestk=bestk)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during
    # training.

    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self,view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)


# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class Gannmodule:
    def __init__(self,ann,index,invariable,insize,outsize, is_output=False):
        self.ann = ann
        self.insize=insize  # Number of neurons feeding into this module
        self.outsize=outsize # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-"+str(self.index)
        self.is_output = is_output
        self.build()

    def build(self):
        mona = self.name; n = self.outsize

        init_weights = self.ann.settings['initial_weight_range']

        self.weights = tf.Variable(np.random.uniform(init_weights[0], init_weights[1], size=(self.insize, n)),
                                   name=mona+'-wgt', trainable=True)  # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(init_weights[0], init_weights[1], size=n),
                                  name=mona+'-bias', trainable=True)  # First bias vector

        if self.is_output:
            # avoid hidden layer activation on output
            self.output = tf.matmul(self.input, self.weights) + self.biases
        else:
            # get hidden layer activation function from settings
            activation = self.ann.settings['hidden_activation_function']

            if activation == 'tanh':
                self.output = tf.nn.tanh(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
            elif activation == 'sigmoid':
                self.output = tf.nn.sigmoid(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
            elif activation == 'leaky_relu':
                self.output = tf.nn.relu(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')
            else:
                # defaults to ReLU
                self.output = tf.nn.relu(tf.matmul(self.input, self.weights) + self.biases, name=mona + '-out')

        self.ann.add_module(self)

    def getvar(self,type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self,type,spec):
        var = self.getvar(type)
        base = self.name +'_'+type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/',var)

# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system

class Caseman():

    def __init__(self,cfunc,vfrac=0,tfrac=0, standardizing=False):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.standardizing = standardizing
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca)  # Randomly shuffle all cases

        if self.standardizing:
            data, _,_ = self.standardize(self.get_input_data(ca))
            targets = self.get_targets(ca)
            self.cases = self.combine_data(data, targets)
        else:
            self.cases = ca

        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = self.cases[0:separator1]
        self.validation_cases = self.cases[separator1:separator2]
        self.testing_cases = self.cases[separator2:]

    def standardize(self, x):
        """Standardize the original data set."""
        mean_x = np.mean(x, axis=0)
        x = x - mean_x
        std_x = np.std(x, axis=0)
        std_x[std_x == 0] = 1
        x = x / std_x
        return x, mean_x, std_x

    def get_input_data(self, data):
        return [data[i][0] for i in range(np.shape(data)[0])]

    def get_targets(self, data):
        return [data[i][1] for i in range(np.shape(data)[0])]

    def combine_data(self, data, targets):
        return [[data[i], targets[i]] for i in range(np.shape(targets)[0])]

    def get_training_cases(self):
        return self.training_cases

    def get_validation_cases(self):
        return self.validation_cases

    def get_testing_cases(self):
        return self.testing_cases


#   ****  MAIN functions ****

def load_poker_dataset(fraction=0.1):
    data = np.loadtxt('data/poker.txt', delimiter=',')

    data_length = np.shape(data)[0]
    reduced_indices = np.random.choice([i for i in range(data_length)], int(fraction * data_length), replace=False)
    data = data[reduced_indices]
    # targets between 0 and 9

    return [[x[:10], TFT.int_to_one_hot(int(x[10]), 10)] for x in data]


def load_wine_dataset():
    data = np.loadtxt('data/winequality_red.txt', delimiter=';')
    # targets are between 3 and 8. Offset left by three to use onehot-encoding
    return [[x[:11], TFT.int_to_one_hot(int(x[11])-3, 6)] for x in data]


def load_yeast_dataset():
    data = np.loadtxt('data/yeast.txt', delimiter=',')
    # targets between 1 and 10
    return [[x[:8], TFT.int_to_one_hot(int(x[8]) - 1, 10)] for x in data]


def load_glass_dataset():
    data = np.loadtxt('data/glass.txt', delimiter=',')
    # targets between 1 and 7, no examples of class 4
    # reducing class labels above 4 by one, to use existing onehot-function
    for i in range(len(data)):
        if data[i][-1] >= 5:
            data[i][-1] -= 1

    return [[x[:9], TFT.int_to_one_hot(int(x[9]) - 1, 6)] for x in data]


def load_mnist(fraction=0.1):
    mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    data_length = len(mnist[0][1])

    reduced_indices = np.random.choice([i for i in range(data_length)], int(fraction * data_length), replace=False)

    data = mnist[0][0][reduced_indices]
    targets = mnist[0][1][reduced_indices]
    data = [i.flatten() for i in data]

    output = [[data[i], TFT.int_to_one_hot(targets[i], 10)] for i in range(len(targets))]
    return output


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def main():
    ################################################
    # Initial dataset choice
    ################################################
    path = 'config/'
    print("Available datasets:")
    for file in os.listdir(path):
        print(os.path.splitext(file)[0])
    print()

    filename = input('Choose a dataset: ')

    with open(path + filename + '.json') as f:
        settings = json.load(f)
    ################################################
    # Edit parameters
    ################################################
    while True:
        print('##########################')
        print('Loaded following settings:')
        print('##########################')
        for key, value in settings.items():
            print('{}: {}'.format(key, value))
        print('##########################')

        print('Enter a parameter name to make changes, or continue by pressing enter.. ')
        print('Separate eventual list values with spaces')
        choice = input('Choice: ')
        if choice in settings.keys():
            new_val = input('New value for ' + choice + ': ')

            # convert list parameters to lists of integers or floats
            if choice in ['network_dimensions', 'initial_weight_range', 'map_layers', 'map_dendrograms', 'display_weights', 'display_biases']:
                new_val = new_val.split(' ')
                new_val = [float(x) if '.' in x else int(x) for x in new_val]
            elif is_integer(new_val):
                new_val = int(new_val)

            settings[choice] = new_val
        elif not choice:
            break
        else:
            print('key not found, try again')
    ################################################
    # Feed parameters to generate dataset and create the GANN
    ################################################
    # Case generator
    if filename == 'autoencoder':
        case_generator = (lambda: TFT.gen_all_one_hot_cases(2**settings['nbits']))
    elif filename == 'dense_autoencoder':
        case_generator = lambda: TFT.gen_dense_autoencoder_cases(settings['case_count'], settings['data_size'], dr=settings['data_range'])
    elif filename == 'bitcounter':
        case_generator = (lambda: TFT.gen_vector_count_cases(settings['ncases'], settings['nbits']))
    elif filename == 'glass':
        case_generator = (lambda: load_glass_dataset())
    elif filename == 'mnist':
        case_generator = (lambda: load_mnist(settings['case_fraction']))
    elif filename == 'parity':
        case_generator = (lambda: TFT.gen_all_parity_cases(settings['nbits']))
    elif filename == 'segmentcounter':
        settings['one_hot'] = True if settings['one_hot'].lower() == 'true' else False
        case_generator = (lambda: TFT.gen_segmented_vector_cases(settings['length'],
                                                                 settings['ncases'],
                                                                 settings['min_seg'],
                                                                 settings['max_seg'],
                                                                 settings['one_hot']))
    elif filename == 'symmetry':
        case_generator = (lambda: TFT.gen_symvect_dataset(settings['nbits'], settings['ncases']))
    elif filename == 'wine':
        case_generator = (lambda: load_wine_dataset())
    elif filename == 'yeast':
        case_generator = (lambda: load_yeast_dataset())
    elif filename == 'poker':
        case_generator = (lambda: load_poker_dataset(settings['case_fraction']))
    else:
        raise ValueError()

    # Case manager
    case_manager = Caseman(cfunc=case_generator,
                           vfrac=settings['validation_fraction'],
                           tfrac=settings['test_fraction'],
                           standardizing=settings['standardize'])

    # Build network
    ann = Gann(dims=settings['network_dimensions'],
               cman=case_manager,
               learning_rate=settings['learning_rate'],
               mbs=settings['minibatch_size'],
               vint=settings['validation_interval'],
               softmax=settings['sm'],
               settings=settings
               )

    ################################################
    # Run the network with provided parameters
    ################################################
    ann.run(epochs=settings['epochs'], bestk=settings['bestk'])

    ################################################
    # create hinton plots from layer activations
    ################################################
    if len(settings['map_layers']) > 0 and settings['map_batch_size'] > 0:
        ann.reopen_current_session()
        ann.do_mapping(settings['map_layers'])
        ann.close_current_session(view=False)

    ################################################
    # create dendrograms from layer activations
    ################################################
    if len(settings['map_dendrograms']) > 0 and settings['map_batch_size'] > 0:
        print('making dendrograms')
        ann.reopen_current_session()
        ann.do_dendrogram(settings['map_dendrograms'])
        ann.close_current_session(view=False)

    ################################################
    # visualize weights and biases for given layers
    ################################################
    if len(settings['display_weights']) > 0:
        ann.reopen_current_session()
        ann.do_wgt_bias_view(settings['display_weights'], type='wgt')
        ann.close_current_session(view=False)

    if len(settings['display_biases']) > 0:
        ann.reopen_current_session()
        ann.do_wgt_bias_view(settings['display_biases'], type='bias')
        ann.close_current_session(view=False)

    PLT.show()


main()

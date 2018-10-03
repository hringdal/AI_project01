import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

dataset = {
    'NETWORK_DIMENSIONS' : [5, 20, 20, 2], # 
    'HIDDEN_ACTIVATION_FUNCTION' : 'relu',
    'OUTPUT_ACTIVATION_FUNCTION' : 'softmax', 
    'COST_FUNCTION' : 'rmse', # RMSE
    'LEARNING_RATE' : 0.001,
    'INITIAL_WEIGHT_RANGE' : [-1, 1],
    'OPTIMIZER' : 'adam',
    'DATA_SOURCE' : None, # data file or function name
    'CASE_FRACTION' : 1.0, # fraction of dataset to use
    'VALIDATION_FRACTION' : 0.1, # fraction of dataset used for validation
    'VALIDATION_INTERVAL' : None, # number of minibatches between each validation test
    'TEST_FRACTION' : 0.1,
    'MINIBATCH_SIZE' : 32,
    'MAP_BATCH_SIZE' : 0, # number of training cases used for a map test. 0 = none
    'EPOCHS' : 100, # number of epochs
    'MAP_LAYERS' : None, # list of layers to visualize during map test
    'MAP_DENDROGRAMS' : None, # list of layers to make dendrograms of
    'DISPLAY_WEIGHTS' : None, # list of weight arrays to display after run
    'DISPLAY_BIASES' : None # list of bias vectors to visualize after run
}

import tflowtools as TFT

class Gann():

    def __init__(self, dims, case_manager, learning_rate=0.1, showint=None, batch_size=10, vint=None, softmax=False):
        self.learning_rate = learning_rate
        self.layer_sizes = dims # Sizes of each layer of neurons, list
        self.show_interval = showint # Frequency of showing grabbed variables
        self.global_training_step = 0 # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []  # Variables to be monitored (by gann code) during a run.
        self.grabvar_figures = [] # One matplotlib figure for each grabvar
        self.minibatch_size = batch_size
        self.validation_interval = vint
        self.validation_history = []
        self.caseman = case_manager
        self.softmax_outputs = softmax
        self.modules = []
        self.build()

    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type, spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self, module_index, type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))
        self.grabvar_figures.append(plt.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self, module):
        self.modules.append(module)

    def build(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input
        insize = num_inputs
        
        # Build all of the modules
        for i, outsize in enumerate(self.layer_sizes[1:]):

            if i == len(self.layer_sizes):
                # avoid relu activation for final output
                gmod = Gannmodule(self, i, invar, insize, outsize, is_output=True)
            else:
                gmod = Gannmodule(self, i, invar, insize, outsize)

            invar = gmod.output
            insize = gmod.outsize
            
        self.output = gmod.output # Output of last module is output of whole network
        
        if self.softmax_outputs:
            # multiclass classification
            self.output = tf.nn.softmax(self.output)
          
        self.target = tf.placeholder(tf.float64,shape=(None,gmod.outsize),name='Target')
        self.configure_learning()

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.

    def configure_learning(self):
        #self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output,
         #                                                                      labels=self.target),
         #                                                                      name='cross-entropy')
        self.error = tf.losses.softmax_cross_entropy(onehot_labels=self.target,
                                                     logits=self.output)
        #tf.sqrt(tf.reduce_mean(tf.square(self.target - self.output)), name='RMSE')
        
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        
        # Defining the training operator
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    def do_training(self, sess, cases, epochs=100, continued=False):
        
        if not continued:
            self.error_history = []
        
        for i in range(epochs):
            # shuffle dataset each epoch
            #np.random.shuffle(cases)
            
            error = 0
            step = self.global_training_step + i
            gvars = [self.error] + self.grabvars  # TRENGER VI DENNE?
            batch_size = self.minibatch_size
            ncases = len(cases)
            nmb = math.ceil(ncases/batch_size)
            
            #for cstart in range(0, ncases, batch_size):  # Loop through cases, one minibatch at a time.
            #    cend = min(ncases, cstart+batch_size)
            #    minibatch = cases[cstart:cend]
            idx = np.random.choice(ncases, batch_size, replace=False)

            cases = np.array(cases)
            minibatch = cases[idx]

            inputs = [c[0] for c in minibatch]
            targets = [c[1] for c in minibatch]
            feeder = {self.input: inputs, self.target: targets}
            _, grabvals, _ = self.run_one_step([self.trainer],
                                               gvars,
                                                self.probes,
                                                session=sess,
                                                feed_dict=feeder,
                                                step=step,
                                                show_interval=self.show_interval)

            error += grabvals[0]
                
            self.error_history.append((step, error))
            self.consider_validation_testing(step,sess)
            
        self.global_training_step += epochs

        TFT.plot_training_history(self.error_history,
                                  self.validation_history,
                                  xtitle="Epoch",
                                  ytitle="Error",
                                  title="History",
                                  fig=True)

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when
    # bestk=None, the standard MSE error function is used for testing.

    def do_testing(self, sess, cases, msg='Testing', bestk=None):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor,
                                                    [TFT.one_hot_to_int(list(v)) for v in targets],
                                                    k=bestk)
            
        testres, grabvals, _ = self.run_one_step(self.test_func,
                                                 self.grabvars,
                                                 self.probes,
                                                 session=sess,
                                                 feed_dict=feeder,
                                                 show_interval=None)
        
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100*(testres/len(cases))))
        
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    def do_mapping(self, sess, cases, msg='Mapping', bestk=None):
        # must reopen current session to work
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.predictor

        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor,
                                                    [TFT.one_hot_to_int(list(v)) for v in targets],
                                                    k=bestk)

        testres, grabvals, _ = self.run_one_step(self.test_func,
                                                 self.grabvars,
                                                 self.probes,
                                                 session=sess,
                                                 feed_dict=feeder,
                                                 show_interval=1)

        print(np.shape(testres))
        print(np.shape(grabvals))

    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns a OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def gen_match_counter(self, logits, labels, k=1):
        """ returns number of correct outputs """
        correct = tf.nn.in_top_k(tf.cast(logits,tf.float32), labels, k)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self, epochs, sess=None, dir="probeview", continued=False):
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.roundup_probes() # this call must come AFTER the session is created, else graph is not in tensorboard.
        self.do_training(session, self.caseman.get_training_cases(), epochs, continued=continued)
    
    def testing_session(self, sess, bestk=None):
        cases = self.caseman.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess, cases, msg='Final Testing', bestk=bestk)

    def mapping_session(self, sess, map_batch_size, bestk=None):
        cases = self.caseman.get_mapping_cases(map_batch_size)
        if len(cases) > 0:
            self.do_mapping(sess, cases, msg='Mapping', bestk=bestk)

    def consider_validation_testing(self, epoch, sess):
        """ Calculate validation error on specified intervals """
        
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.caseman.get_validation_cases()
            
            if len(cases) > 0:
                error = self.do_testing(sess, cases, msg='Validation')
                self.validation_history.append((epoch, error))

    def test_on_trains(self, sess, bestk=None):
        """ Do testing (i.e. calc error without learning) on the training set. """
        
        self.do_testing(sess, self.caseman.get_training_cases(), msg='Total Training', bestk=bestk)

    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                     session=None, feed_dict=None, step=1, show_interval=1):
        """ Similar to the "quickrun" functions used earlier. """
        
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars],
                               feed_dict=feed_dict)
            
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        
        return results[0], results[1], sess

    def run_mapping_step(self, ):
        operators = []
        result = self.current_session.run([operators, [self.modules[0].weights, self.modules[1].weights], [self.modules[0].bias], [self.modules[0].activation]], feed_dict=feed_dict)

    # bullshit
    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars]
        msg = "Grabbed Variables at Step " + str(step)
        print('\n' + msg, end='\n')
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if names:
                print("   " + names[i] + " = ", end="")
                
            if type(v) == np.ndarray and len(v.shape) > 1: # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v,
                                fig=self.grabvar_figures[fig_index],
                                title= names[i]+ ' at step '+ str(step))
                fig_index += 1
            else:
                print(v, end="\n")

    def run(self, epochs=100, sess=None, continued=False, bestk=None):
        self.training_session(epochs, sess=sess, continued=continued)
        self.test_on_trains(sess=self.current_session, bestk=bestk)
        self.testing_session(sess=self.current_session, bestk=bestk)
        self.close_current_session(view=False)

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self, epochs=100, bestk=None):
        self.reopen_current_session()
        self.run(epochs, sess=self.current_session, continued=True, bestk=bestk)

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
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard instance
        self.current_session.run(tf.global_variables_initializer()) # initialize all variables
        self.restore_session_params()  # Reload old weights and biases to continue from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self,view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)


class Gannmodule:

    def __init__(self, ann, index, in_variable, insize, outsize, is_output=False):
        self.ann = ann
        self.insize = insize  # Number of neurons feeding into this module
        self.outsize = outsize # Number of neurons in this module
        self.input = in_variable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-" + str(self.index)
        self.is_output = is_output
        self.build()

    def build(self):
        mona = self.name
        n = self.outsize

        self.weights = tf.Variable(np.random.uniform(-.1, .1, size=(self.insize, n)),
                                   name = mona + '-wgt',
                                   trainable = True) # True = default for trainable anyway
        
        self.biases = tf.Variable(np.random.uniform(-.1, .1, size=n),
                                  name = mona + '-bias',
                                  trainable = True)  # First bias vector

        if self.is_output:
            self.output = tf.matmul(self.input, self.weights) + self.biases
        else:
            self.output = tf.nn.relu(tf.matmul(self.input, self.weights)+self.biases, name=mona + '-out')

        self.ann.add_module(self)

    def getvar(self, type):
        # type either (in, out, wgt, bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    def gen_probe(self, type, spec):
        # spec: a list, can contain one or more of (avg,max,min,hist)
        # type either (in, out, wgt, bias)
        var = self.getvar(type)
        base = self.name + '_' + type
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


class CaseManager:
    def __init__(self, cfunc, vfrac=0, tfrac=0, standardizing=False):
        self.standardizing = standardizing
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]
        print(np.shape(self.cases))

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca) # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases) * self.validation_fraction)
        self.training_cases = ca[:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]
        
        if self.standardizing:
            train_data, train_mean, train_std = self.standardize(self.get_input_data(self.training_cases))
            validation_data = self.standardize_test(self.get_input_data(self.validation_cases), train_mean, train_std)
            test_data = self.standardize_test(self.get_input_data(self.testing_cases), train_mean, train_std)
            
            train_targets = self.get_targets(self.training_cases)
            validation_targets = self.get_targets(self.validation_cases)
            test_targets = self.get_targets(self.testing_cases)
          
            self.training_cases = self.combine_data(train_data, train_targets)
            self.validation_cases = self.combine_data(validation_data, validation_targets)
            self.test_data = self.combine_data(test_data, test_targets)
          

    def get_training_cases(self):
        return self.training_cases
    
    def get_validation_cases(self):
        return self.validation_cases
    
    def get_testing_cases(self):
        return self.testing_cases

    def get_mapping_cases(self, n):
        return self.training_cases[:n]
      
    def get_input_data(self, data):
        return [data[i][0] for i in range(np.shape(data)[0])]
      
    def get_targets(self, data):
        return [data[i][1] for i in range(np.shape(data)[0])]
      
    def combine_data(self, data, targets):
        return [[data[i], targets[i]] for i in range(np.shape(targets)[0])]
      
    def standardize(self, x):
        """Standardize the original data set."""
        mean_x = np.mean(x, axis=0)
        x = x - mean_x
        std_x = np.std(x, axis=0)
        std_x[std_x == 0] = 1
        x = x / std_x
        return x, mean_x, std_x

    def standardize_test(self, x, mean_x, std_x):
        """Standarize the test data with the same mean and stddev as the training set"""
        x = (x - mean_x) / std_x
        return x

def load_wine_dataset():
    data = np.loadtxt('data/winequality_red.txt', delimiter=';')
    # targets are between 3 and 8. Offset left by three to use onehot-encoding
    return [[x[:11], TFT.int_to_one_hot(int(x[11])-3, 6)] for x in data]


def load_yeast_dataset():
    data = np.loadtxt('data/yeast.txt', delimiter=',')
    # targets between 1 and 10
    return [[x[:8], TFT.int_to_one_hot(int(x[8])-1, 10)] for x in data]


def load_glass_dataset():
    data = np.loadtxt('data/glass.txt', delimiter=',')
    # targets between 1 and 7, no examples of class 4
    # reducing class labels above 4 by one, to use existing onehot-function
    for i in range(len(data)):
        if data[i][-1] >= 5:
            data[i][-1] -= 1
    
    return [[x[:9], TFT.int_to_one_hot(int(x[9])-1, 6)] for x in data]


def load_mnist(fraction=0.1):
    mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    data_length = len(mnist[0][1])
    
    reduced_indices = np.random.choice([i for i in range(data_length)], int(fraction*data_length), replace=False)
    
    data = mnist[0][0][reduced_indices]
    targets = mnist[0][1][reduced_indices]
    data = [i.flatten() for i in data]
    
    output = [[data[i], TFT.int_to_one_hot(targets[i], 10)] for i in range(len(targets))]
    return output


def autoex(epochs=300, nbits=4, learning_rate=0.03, showint=100, batch_size=None, vfrac=0.1, tfrac=0.1, vint=100, sm=False, bestk=None):
    size = 2**nbits
    batch_size = batch_size if batch_size else size
    case_generator = (lambda : TFT.gen_all_one_hot_cases(2**nbits))
    case_manager = CaseManager(cfunc=case_generator,vfrac=vfrac,tfrac=tfrac)
    ann = Gann(dims=[size,nbits,size],case_manager=case_manager,learning_rate=learning_rate,showint=showint,batch_size=batch_size,vint=vint,softmax=sm)
    #ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    #ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    #ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs, bestk=bestk)
    ann.runmore(epochs*2, bestk=bestk)
    return ann

def number_of_1s(epochs=6000, nbits=15, ncases=500, learning_rate=0.05, showint=None, batch_size=32, vfrac=0.1, tfrac=0.1, vint=200, sm=True, bestk=1):
    case_generator = (lambda: TFT.gen_vector_count_cases(ncases, nbits))
    case_manager = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = Gann(dims=[nbits, nbits*3, nbits+1], case_manager=case_manager, learning_rate=learning_rate, showint=showint, batch_size=batch_size, vint=vint, softmax=sm)
    ann.run(epochs,bestk=bestk)
    plt.show()
    return ann
  

def parity(epochs=2000, nbits=10, learning_rate=0.001, showint=None, batch_size=256, vfrac=0.1, tfrac=0.1, vint=200, sm=True, bestk=1):
    case_generator = (lambda: TFT.gen_all_parity_cases(nbits))
    case_manager = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = Gann(dims=[nbits, 100, 100, 2], case_manager=case_manager, learning_rate=learning_rate, showint=showint, batch_size=batch_size, vint=vint, softmax=sm)
    ann.run(epochs, bestk=bestk)
    plt.show()
    return ann

  
def symmetry(epochs=3000, nbits=101, ncases=2000, learning_rate=0.001, showint=1000, batch_size=512, vfrac=0.1, tfrac=0.1, vint=200, sm=True, bestk=1):
    case_generator = (lambda: TFT.gen_symvect_dataset(nbits, ncases))
    case_manager = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = Gann(dims=[nbits, nbits*3, 2], case_manager=case_manager, learning_rate=learning_rate, showint=showint, batch_size=batch_size, vint=vint, softmax=sm)
    ann.run(epochs,bestk=bestk)
    # TFT.fireup_tensorboard('probeview')
    return ann
  

def segments(epochs=3001, length=25, ncases=1000, min_seg=0, max_seg=8, one_hot=True, learning_rate=0.0001, showint=1000, batch_size=32, vfrac=0.1, tfrac=0.1, vint=200, sm=True, bestk=1):
    case_generator = (lambda: TFT.gen_segmented_vector_cases(length, ncases, min_seg, max_seg, one_hot))
    case_manager = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = Gann(dims=[length, 500, max_seg-min_seg+1], case_manager=case_manager, learning_rate=learning_rate, showint=showint, batch_size=batch_size, vint=vint, softmax=sm)
    ann.run(epochs,bestk=bestk)
    # TFT.fireup_tensorboard('probeview')
    return ann


def wine(epochs=10000, learning_rate=0.00001, showint=0, batch_size=1024, vfrac=0.1, tfrac=0.1, vint=500, sm=True, bestk=1):
    case_generator = (lambda: load_wine_dataset())
    case_manager = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac, standardizing=True)
    ann = Gann(dims=[11, 128, 128, 6], case_manager=case_manager, learning_rate=learning_rate, showint=showint, batch_size=batch_size, vint=vint, softmax=sm)
    ann.run(epochs,bestk=bestk)

    
def yeast(epochs=5000, learning_rate=0.001, showint=0, batch_size=512, vfrac=0.1, tfrac=0.1, vint=1000, sm=True, bestk=1):
    case_generator = (lambda: load_yeast_dataset())
    case_manager = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac, standardizing=True)
    ann = Gann(dims=[8, 200, 100, 10], case_manager=case_manager, learning_rate=learning_rate, showint=showint, batch_size=batch_size, vint=vint, softmax=sm)
    ann.run(epochs,bestk=bestk)
    plt.show()


def glass(epochs=10000, learning_rate=0.0001, showint=0, batch_size=128, vfrac=0.1, tfrac=0.1, vint=200, sm=True, bestk=1):
    case_generator = (lambda: load_glass_dataset())
    case_manager = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac, standardizing=True)
    ann = Gann(dims=[9, 100, 100, 100, 100, 6], case_manager=case_manager, learning_rate=learning_rate, showint=showint, batch_size=batch_size, vint=vint, softmax=sm)
    ann.run(epochs,bestk=bestk)
    plt.show()


def MNIST(epochs=2000, learning_rate=0.001, showint=0, batch_size=64, data_frac=0.1, vfrac=0.1, tfrac=0.1, vint=40, sm=True, bestk=1):
    case_generator = (lambda: load_mnist(data_frac))
    case_manager = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac, standardizing=True)
    ann = Gann(dims=[784, 150, 10], case_manager=case_manager, learning_rate=learning_rate, showint=showint, batch_size=batch_size, vint=vint, softmax=sm)
    ann.run(epochs,bestk=bestk)
    # TFT.fireup_tensorboard('probeview')
    return ann

number_of_1s()
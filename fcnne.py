import tensorflow as tf
import numpy as np
import csv
import os
import argparse

__author__ = 'Arseny Skryagin'


def m_function(model):
    """
    Derive the function for the choosen model m_{i} (i = 1,2,3,4,5).

    Args_____________________________________________________________________
            model number i from {1,2,3,4,5}

    Returns______________________________________________________________________
        a function m_{i}(x::<class 'float'>)
        E.g.: <function m_function.<locals>.m_1 at 0x7f96f711bbf8>
    """
    if model == 1:
        def m_1(x):
            return 1/np.tan(np.pi/(1 + np.exp(x[0]**2 + 2*x[1] + np.sin(6*(x[3]**3)) - 3)))\
                   + np.exp(3*x[2] + 2*x[3] - 5*x[4] + np.sqrt(x[5] + 0.9*x[6] + 0.1))
        return m_1
    elif model == 2:
        def m_2(x):
            return 2/(x[0] + 0.008) + 3*(np.log(x[1]**7 * x[2] + 0.1))*x[3]
        return m_2
    elif model == 3:
        def m_3(x):
            return 2*np.log(x[0]*x[1] + 4*x[2] + np.abs(np.tan(x[3]))) \
                   + (x[2]**4)*(x[4]**2)*x[5] - x[3]*x[6] \
                   + (3*x[7]**2 + x[8] + 2)**(0.1 + 4*x[9]**2)
        return m_3
    elif model == 4:
        def m_4(x):
            return x[0] + np.tan(x[1]) + x[2]**3 + np.log(x[3]) + 3*x[4] \
                   + x[5] + np.sqrt(x[6])
        return m_4
    elif model == 5:
        def m_5(x):
            return np.exp(np.linalg.norm(x))
        return m_5
    else:
        return False


def generate_data(n_learn, n_test, n_evaluate, sigma, model):
    """
    Generate data points for the choosen model.

    The following vectors will be sampled
                    X ~ U([0,1]^d)
                    epsilon ~ N((mu=0.0, sigma=1.0))
                    Y = m(X) + sigma*lambda*epsilon

    Args_____________________________________________________________________
            n_learn - a size (i.e. a natural number (<class 'int'>)
                      of the dataset for training
            n_test - a size (i.e. a natural number (<class 'int'>))
                     of the dataset for test
            n_evaluate - a size (i.e. a natural number (<class 'int'>))
                         of the dataset for evaluation
            sigma - the real number (i.e. <class 'float'>) sigma
                    (scattering factor)
            model - a natural number (i.e. <class 'int'>)
                    (model number aka choosen model)

     Returns_____________________________________________________________________
            d (<class 'int'>) - dimension of x_* datasets
            x_learn, y_learn (<class 'numpy.ndarray'>)- a learn dataset
            x_test, y_test (<class 'numpy.ndarray'>) - a test dataset
    """
    if model < 1 or model > 5:
        print('The model number {} is wrong!'.format(model))
        return '''Error! There is no model for i={} in 'm_function'
                  defined.'''.format(model)
    d = 10 if model == 3 else 7  # Set dimension depending on the chosen model.
    m = m_function(model)  # Get the map function for the chosen model.
    lambdas = [9.11, 5.68, 13.97, 1.94, 1.64]  # An array of the 'lambdas'
    # Chose the value of lambda dependening on the model number.
    vlambda = lambdas[model - 1]
    # Generate the learning dataset.
    x_learn = np.random.rand(n_learn, d)
    epsilon_learn = np.random.randn(n_learn)
    y_learn = np.zeros([n_learn, 1])
    for i in np.arange(n_learn):
        y_learn[i, :] = m(x_learn[i, :]) + sigma*vlambda*epsilon_learn[i]
    # Generate the test dataset.
    x_test = np.random.rand(n_test, d)
    epsilon_test = np.random.randn(n_test)
    y_test = np.zeros([n_test, 1])
    for i in np.arange(n_test):
        y_test[i, :] = m(x_test[i, :]) + sigma*vlambda*epsilon_test[i]
    # Generate the evaluation dataset.
    x_evaluate = np.random.rand(n_evaluate, d)
    y_evaluate = np.zeros([n_evaluate, 1])
    for i in np.arange(n_evaluate):
        y_evaluate[i, :] = m(x_evaluate[i, :])
    return d, x_learn, y_learn, x_test, y_test, x_evaluate, y_evaluate


def construct_a_MLF(d, L, r, actfunc, initialization):
    """
    Construct a Multi-Layer Feedforward Network (MLF).

    Args_____________________________________________________________________
                      d (<class 'int'>) - dimension of x_* datasets
                      batch_size (<class 'int'>) - batch size as a natural
                                                   number
                      L - (<class 'int'>) number of the hidden layers
                      r - (<class 'int'>) number of the nodes in each layer

     Returns_____________________________________________________________________
                       weight_matrices (<class 'list'>) - a list of weight
                            matrices for all layers in the neural network
                      bias_vectors (<class 'list'>) - a list of bias vectors
                            for all layers in the neural network
                      layers (<class 'list'>) - an array of hidden layers
                            including the first (or input) layer
    """
    # Define place holders X and Y for the given data set.
    X = tf.placeholder(tf.float64, shape=[None, d], name='X')
    Y_obs = tf.placeholder(tf.float64, shape=[None, 1], name='Y_obs')
    # Create the list for weight matrices.
    weight_matrices = []
    # Create the list for bias vectors.
    bias_vectors = []
    # Create the list for layers of the neural network to construct.
    layer = []
    # Initialize weights and biases according to
    # the chosen initialization scheme.
    if initialization == 'small':
        mu, sigma = 0, 0.001
        # First layer
        weight_matrices.append(tf.get_variable('W_in', dtype=tf.float64,
                                               initializer=np.random.normal(mu, sigma, [d, r])))
        bias_vectors.append(tf.get_variable('B_in', dtype=tf.float64,
                                            initializer=np.random.normal(mu, sigma, r)))
        # Add hidden layers.
        for i in range(1, L+1):
            weight_matrices.append(tf.get_variable('W_{}'.format(i),
                                                   dtype=tf.float64,
                                                   initializer=np.random.normal(mu, sigma, [r, r])))
            bias_vectors.append(tf.get_variable('B_{}'.format(i),
                                                dtype=tf.float64,
                                                initializer=np.random.normal(mu, sigma, r)))
        # Add final layer.
        weight_matrices.append(tf.get_variable('W_out', dtype=tf.float64,
                                               initializer=np.random.normal(mu, sigma, [r, 1])))
        bias_vectors.append(tf.get_variable('B_out', dtype=tf.float64,
                                            initializer=np.random.normal(mu, sigma, 1)))
    elif actfunc == 'sigmoid' and initialization == 'Xavier':
        weight_matrices.append(tf.get_variable('W_in', dtype=tf.float64,
                                               initializer=np.random.randn(d, r)*np.sqrt(2/(d+r))))
        bias_vectors.append(tf.get_variable('B_in', dtype=tf.float64,
                                            initializer=np.zeros(r)))
        for i in range(1, L+1):
            weight_matrices.append(tf.get_variable('W_{}'.format(i),
                                                   dtype=tf.float64,
                                                   initializer=np.random.randn(r, r)*np.sqrt(1/r)))
            bias_vectors.append(tf.get_variable('B_{}'.format(i),
                                                dtype=tf.float64,
                                                initializer=np.zeros(r)))
        weight_matrices.append(tf.get_variable('W_out', dtype=tf.float64,
                                               initializer=np.random.randn(r, 1)*np.sqrt(2/(r+1))))
        bias_vectors.append(tf.get_variable('B_out', dtype=tf.float64,
                                            initializer=np.zeros(1)))
    elif actfunc == 'ReLU' and initialization == 'He':
        weight_matrices.append(tf.get_variable('W_in', dtype=tf.float64,
                                               initializer=np.random.randn(d, r)*np.sqrt(2/d)))
        bias_vectors.append(tf.get_variable('B_in', dtype=tf.float64,
                                            initializer=np.zeros(r)))
        for i in range(1, L+1):
            weight_matrices.append(tf.get_variable('W_{}'.format(i),
                                                   dtype=tf.float64,
                                                   initializer=np.random.randn(r, r)*np.sqrt(2/r)))
            bias_vectors.append(tf.get_variable('B_{}'.format(i),
                                                dtype=tf.float64,
                                                initializer=np.zeros(r)))
        weight_matrices.append(tf.get_variable('W_out', dtype=tf.float64,
                                               initializer=np.random.randn(r, 1)*np.sqrt(2/r)))
        bias_vectors.append(tf.get_variable('B_out', dtype=tf.float64,
                                            initializer=np.zeros(1)))
    elif initialization == 'Schmidt-Hieber':
        mu, sigma = 0, 1
        weight_matrices.append(tf.get_variable('W_in',
                                               dtype=tf.float64,
                                               initializer=np.clip(np.random.normal(mu, sigma, [d, r]), -1, 1)))
        bias_vectors.append(tf.get_variable('B_in',
                                            dtype=tf.float64,
                                            initializer=np.clip(np.random.normal(mu, sigma, r), -1, 1)))
        for i in range(1, L+1):
            weight_matrices.append(tf.get_variable('W_{}'.format(i),
                                                   dtype=tf.float64,
                                                   initializer=np.clip(np.random.normal(mu, sigma, [r, r]), -1, 1)))
            bias_vectors.append(tf.get_variable('B_{}'.format(i),
                                                dtype=tf.float64,
                                                initializer=np.clip(np.random.normal(mu, sigma, r), -1, 1)))
        weight_matrices.append(tf.get_variable('W_out', dtype=tf.float64,
                                               initializer=np.clip(np.random.normal(mu, sigma, [r, 1]), -1, 1)))
        bias_vectors.append(tf.get_variable('B_out', dtype=tf.float64,
                                            initializer=np.clip(np.random.normal(mu, sigma, 1), -1, 1)))
    elif actfunc == 'tanh' and initialization == 'Xavier':
        weight_matrices.append(tf.get_variable('W_in', dtype=tf.float64,
                                               initializer=np.random.randn(d, r)*np.sqrt(1/d)))
        bias_vectors.append(tf.get_variable('B_in', dtype=tf.float64,
                                            initializer=np.zeros(r)))
        for i in range(1, L+1):
            weight_matrices.append(tf.get_variable('W_{}'.format(i),
                                                   dtype=tf.float64,
                                                   initializer=np.random.randn(r, r)*np.sqrt(1/r)))
            bias_vectors.append(tf.get_variable('B_{}'.format(i),
                                                dtype=tf.float64,
                                                initializer=np.zeros(r)))
        weight_matrices.append(tf.get_variable('W_out', dtype=tf.float64,
                                               initializer=np.random.randn(r, 1)*np.sqrt(1/r)))
        bias_vectors.append(tf.get_variable('B_out', dtype=tf.float64,
                                            initializer=np.zeros(1)))
    # Define model.
    if actfunc == 'ReLU':
        # Construct input layer.
        layer.append(tf.nn.relu(tf.add(tf.matmul(X, weight_matrices[0]),
                                       bias_vectors[0])))
        # Add hidden layers.
        for i in range(1, L+2):
            layer.append(tf.nn.relu(tf.add(tf.matmul(layer[i-1], weight_matrices[i]),
                                           bias_vectors[i])))
        # Construct the output layer.
        Y = tf.add(tf.matmul(layer[L], weight_matrices[L+1]),
                   bias_vectors[L+1], name='out_{}_{}'.format(L, r))
    elif actfunc == 'sigmoid':
        layer.append(tf.nn.sigmoid(tf.add(tf.matmul(X, weight_matrices[0]),
                                          bias_vectors[0])))
        for i in range(1, L+2):
            layer.append(tf.nn.sigmoid(tf.add(tf.matmul(layer[i-1], weight_matrices[i]),
                                              bias_vectors[i])))
        Y = tf.add(tf.matmul(layer[L], weight_matrices[L+1]),
                   bias_vectors[L+1], name='out_{}_{}'.format(L, r))
    elif actfunc == 'tanh':
        layer.append(tf.nn.tanh(tf.add(tf.matmul(X, weight_matrices[0]),
                                       bias_vectors[0])))
        for i in range(1, L+2):
            layer.append(tf.nn.tanh(tf.add(tf.matmul(layer[i-1], weight_matrices[i]),
                                           bias_vectors[i])))
        Y = tf.add(tf.matmul(layer[L], weight_matrices[L+1]),
                   bias_vectors[L+1], name='out_{}_{}'.format(L, r))
    return X, Y_obs, Y


def compute_L2_error(batch_size, x_evaluate, y_evaluate, session,
                     loss, X, Y_obs):
    """
    Compute a L2 error for the given x_evaluate and y_evaluate.

    Args_____________________________________________________________________
            batch_size - <class 'int'> or <class 'bool'> the size of the batch
            x_evaluate, y_evalue (<class 'numpy.ndarray'>)- a evaluate dataset
            session - <class 'tensorflow.python.client.session.Session'>
                the instance of the Session-object for the specified
                computational graph
            loss - a defined loss function. E.g. MSE - the mean squared error
            X - a placeholder for a batch of data (i.e. batch of x_evaluate)
            Y_obs - a placeholder  for a batch of Y (i.e. batch of y_evaluate)

     Returns_____________________________________________________________________
            L2_error <class 'numpy.float64'>
    """
    L2_error = 0.0
    n_evaluate = np.size(x_evaluate, 0)
    if not batch_size:
        batch_errors = []
        for i in np.arange(int(np.size(x_evaluate, 0)/batch_size)):
            loss_error = 0
            batch_x = x_evaluate[i*batch_size:(i+1)*batch_size, :]
            batch_y = y_evaluate[i*batch_size:(i+1)*batch_size, :]
            loss_error = session.run(loss, feed_dict={X: batch_x,
                                                      Y_obs: batch_y})
            batch_errors.append(loss_error)
        L2_error = np.sum(batch_errors)/n_evaluate
        del(batch_errors)
    else:
        loss_error = session.run(loss, feed_dict={X: x_evaluate,
                                                  Y_obs: y_evaluate})
        L2_error = loss_error/np.size(y_evaluate, 0)
    print('L2 error is {}'.format(L2_error))
    return L2_error


def fc_nn_estimate(d, batch_size, x_learn, y_learn, x_test, y_test,
                   x_evaluate, y_evaluate, actfunc, initialization,
                   L_list, r_list, epochs):
    """
    Fully connected neural network estimate.

    This function chose adaptive the number of hidden layers 'L' and the number
    of the nodes 'r' for each layer, trains in accordance with these numbers
    a Multi-Layer Feedforward Networks (MLF), and test those against each other
    in order to find one which fits the given learning data set at best.

    Args_____________________________________________________________________
        d (<class 'int'>) - dimension of x_* datasets
        batch_size (<class 'int'>) - batch size as a natural number
        x_learn, y_learn (<class 'numpy.ndarray'>)- a learn dataset
        x_test, y_test (<class 'numpy.ndarray'>) - a test dataset
        actfunc (<class 'str'>) - type of activation function, i.e 'ReLU',
                                  'tanh' or 'sigmoid'
        L_list (<class 'list'>) - a list of 'L' possible numbers of
                                  hidden layers
        r_list (<class 'list'>) - a list of 'r' possible nodes each
                                  hidden layer
        epochs (<class 'int'>)  - a number of training epochs

    Returns_____________________________________________________________________
    the dictionary which contains a MLF with the smallest L2 error.
    I.e..: <class 'dict'> with four entries:
        'Validation error' : <class 'float'>
        'L_r'              : <class 'tuple'>
        'L2_error'         : <class 'float'>
    """
    n_learn = np.size(x_learn, 0)
    n_test = np.size(x_test, 0)
    graphs = []
    for L in L_list:
        for r in r_list:
            g = tf.Graph()
            with g.as_default():
                X, Y_obs, Y = construct_a_MLF(d, L, r, actfunc, initialization)
                # Define the loss function which is used for all phases,
                # i.e. for training, validation and evaluation phases alike.
                loss = tf.reduce_sum(tf.squared_difference(Y, Y_obs),
                                     name='loss_func_{}_{}'.format(L, r))
                # Optimizing function for back propagation
                train_op = tf.train.AdamOptimizer(name='Adam_{}_{}'
                                                  .format(L, r)).minimize(loss)
                # Create the TensorFlow-'Session' with constructed graph.
                session = tf.Session(graph=g)
                # Define all variables inside of the 'Session'.
                session.run(tf.initializers.global_variables())
                # Training phase
                print('Begin the training phase of a MLF with (L, r) = ({}, {})'
                      .format(L, r))
                for epoch in np.arange(1, epochs+1):
                    if not batch_size:
                        total_error = 0.0
                        for i in np.arange(int(n_learn/batch_size)):
                            batch_x = x_learn[i*batch_size:(i+1)*batch_size, :]
                            batch_y = y_learn[i*batch_size:(i+1)*batch_size, :]
                            batch_loss, _ = session.run([loss, train_op],
                                                        feed_dict={X: batch_x,
                                                                   Y_obs: batch_y})
                            total_error += batch_loss
                        total_error /= n_learn
                        if epoch == 1:
                            print('For the first epoch is the loss {}.'
                                  .format(total_error))
                        elif epoch == epochs:
                            print('For the last epoch is the loss {}.'
                                  .format(total_error))
                        del total_error
                    else:
                        total_loss, _ = session.run([loss, train_op],
                                                    feed_dict={X: x_learn,
                                                               Y_obs: y_learn})
                        total_loss /= n_learn
                        if epoch == 1:
                            print('For the first epoch is the loss {}.'
                                  .format(total_loss))
                        elif epoch == epochs:
                            print('For the last epoch is the loss {}.'
                                  .format(total_loss))
                        del total_loss
                # Compute validation error.
                validation_error = session.run(loss, feed_dict={X: x_test,
                                                                Y_obs: y_test})
                validation_error /= n_test
                print('Validation error is {}'.format(validation_error))
                l2_error = compute_L2_error(batch_size, x_evaluate, y_evaluate,
                                            session, loss, X, Y_obs)
                graphs.append({'L_r': (L, r),
                               'Validation error': validation_error,
                               'L2_error': l2_error})
                del validation_error
                session.close()
            tf.reset_default_graph()
    opt_model = sorted(graphs, key=lambda k: k['L2_error'])[0]
    del graphs
    return opt_model


def sim(n_learn, n_test, n_evaluate, batch_size, sigma, model,
        actfunc, initialization, L_list, r_list, epochs):
    """
    Singe simulation.

    This function performs a single 'simulation'

    Args_____________________________________________________________________
        n_learn - a size (i.e. a natural number (<class 'int'>)) of
                  the dataset for training
        n_test - a size (i.e. a natural number (<class 'int'>)) of
                 the dataset for test
        n_evaluate - a size (i.e. a natural number (<class 'int'>)) of
                     the dataset for evaluation
        batch_size - a size (i.e. a natural number (<class 'int'>)) of
                     batch if in-batch training was chosen. It might
                     be also <class 'bool'> if no-batch modus was chosen
        sigma - the real number (i.e. <class 'float'>) sigma
                (scattering factor)
        model - a natural number (i.e. <class 'int'>) (chosen model number)
        actfunc - the chosen activation function (i.e. either sigmoid, tanh
                  or ReLU (<class 'str'>))
        initialization - the chosen initialization scheme (i.e. <class 'str'>)
        L_list - an array of natural numbers (i.e. <class 'numpy.ndarray'>)
                 which contains all possible amounts of hidden layers
        r_list - an array of natural numbers (i.e. <class 'numpy.ndarray'>)
                 which contains all possible amounts of nodes for each
                 layer in a MLF
        epochs - a natural number (i.e. <class 'int'>) of training iterations
    Returns_____________________________________________________________________
        L2_error - a real number (i.e. <class 'float'>) of the mean
                   squared error (aka MSE)
    """
    # Generate data for a single simulation.
    d, x_learn, y_learn, x_test, y_test, x_evaluate, y_evaluate = generate_data(
                                      n_learn, n_test, n_evaluate, sigma, model)
    # Obtain a optimal model for the generated datasets.
    opt_model = fc_nn_estimate(d, batch_size, x_learn, y_learn, x_test, y_test,
                               x_evaluate, y_evaluate, actfunc, initialization,
                               L_list, r_list, epochs)
    L = opt_model['L_r'][0]
    r = opt_model['L_r'][1]
    L2_error = opt_model['L2_error']
    return L2_error, L, r


def seq_sim(repititions, batch_size, n_learn, n_test, n_evaluate, sigma, model,
            actfunc, initialization, L_list, r_list, epochs, repmedian):
    """
    Sequence of simulations build upon of single simulation.

    This function perform a sequence of 'simulations', compute the median
    as well as the interquartile range (IQR) and save all of the data in
    a single CSV file

    Args_____________________________________________________________________
        repititions - a natural number (i.e. <class 'int'>) number of
                      'simulations' to perform
        n_learn - a size (i.e. a natural number (<class 'int'>)) of the dataset
                  for training
        n_test - a size (i.e. a natural number (<class 'int'>)) of the dataset
                 for test
        n_evaluate - a size (i.e. a natural number (<class 'int'>)) of the
                     dataset for evaluation
        sigma - the real number (i.e. <class 'float'>) sigma
                (scattering factor)
        model - a natural number (i.e. <class 'int'>) (chosen model number)
        actfunc - the chosen activation function (i.e. either sigmoid, tanh
                  or ReLU (<class 'str'>))
        L_list - an array of natural numbers (i.e. <class 'list'>) which
                 contains all possible numbers of hidden layers
        r_list - an array of natural numbers (i.e. <class 'list'>) which
                 contains all possible numbers of nodes for each layer in a MLF
        epochs - a natural number (i.e. <class 'int'>) of training iterations
        repmedian - a natural number (i.e. <class 'int'>) of runs to generate
                    'enough' data (i.e. in accordance with 'repititions') which
                    are used to compute the error normalization
    """
    n = n_learn + n_test
    # Force the programm to use only second physical core of the processor.
    os.system('taskset -cp {} {}'.format(1, os.getppid()))
    # Make certain the order structure is existing. If not,
    # create the non-existing parts of the structure.
    order_path = ''.join([os.getcwd(),
                          '/model_{}/act_func_{}/initialization_{}/sigma_{}_procent/n_{}'
                          .format(model, actfunc, initialization, int(100*sigma), n)])
    if not os.path.exists(order_path):
        os.makedirs(order_path)
    file_name = 'results_model_{}.csv'.format(model)
    # Make certain that the according CSV-file exists.
    if not os.path.isfile(file_name):
        csv_file = open(file_name, mode='w')
        writer = csv.DictWriter(csv_file,
                                fieldnames=['run', 'error', 'L', 'r'])
        writer.writeheader()
        csv_file.close()
    print('Compute normalization of error.')
    error_list = np.zeros(repmedian)
    for i in np.arange(repmedian):
        d, x_learn, y_learn, x_test, y_test, x_evaluate, y_evaluate = \
                        generate_data(1, 1, n_evaluate, sigma, model)
        est = np.mean(y_evaluate)
        error_val = 0.0
        for j in np.arange(n_evaluate):
            error_val += (y_evaluate[j] - est)**2
        error_list[i] = error_val/n_evaluate
    error_normalization = np.median(error_list)
    print('The normalization of error is {}'.format(error_normalization))
    # Save error_normalization in a separeted CSV-file.
    error_normalization_file = 'error_normalization_model_{}.csv'.format(model)
    if not os.path.isfile(error_normalization_file):
        csv_file = open(error_normalization_file, mode='w')
        writer = csv.DictWriter(csv_file, fieldnames=['error_normalization'])
        writer.writeheader()
        writer.writerow({'error_normalization': error_normalization})
        csv_file.close()
    else:
        csv_file = open(error_normalization_file, mode='a')
        writer = csv.DictWriter(csv_file, fieldnames=['error_normalization'])
        writer.writerow({'error_normalization': error_normalization})
        csv_file.close()
    del error_list
    error_list = np.zeros(repititions)
    for i in np.arange(repititions):
        print('Compute estimate number {}'.format(i + 1))
        l2_error, opt_L, opt_r = sim(n_learn, n_test, batch_size, n_evaluate,
                                     sigma, model, actfunc, initialization,
                                     L_list, r_list, epochs)
        error_list[i] = l2_error/error_normalization
        csv_file = open(file_name, mode='a')
        writer = csv.DictWriter(csv_file, fieldnames=['run', 'error',
                                                      'L', 'r'])
        writer.writerow({'run': i+1, 'error': error_list[i],
                         'L': opt_L, 'r': opt_r})
        csv_file.close()
    del error_list
    # Check if any simulations are already performed.
    count = 0
    if os.path.isfile(file_name):
        csv_file = open(file_name, mode='r')
        reader = csv.reader(csv_file)
        # Skip the header of the CSV file.
        next(reader)
        for row in reader:
            count += 1
        csv_file.close()
    # If 50 simulations are performed, then read 'error' colum and compute the
    # median and the IQR. Both numbers will be appended to the csv file.
    if count == 50:
        csv_file = open(file_name, mode='r')
        reader = csv.reader(csv_file)
        # Skip the header of the CSV file.
        next(reader)
        error_list = []
        for row in reader:
            error_list.append(float(row[1]))
        csv_file.close()
        # Calculate the median and the IQR.
        median_of_error_list = np.median(error_list)
        print('Median error is {}'.format(median_of_error_list))
        iqr_of_error_list = np.subtract(*np.percentile(error_list, [75, 25]))
        print('IQR is {}'.format(iqr_of_error_list))
        csv_file = open(file_name, mode='a')
        writer = csv.DictWriter(csv_file, fieldnames=['Median', 'IQR'])
        writer.writeheader()
        writer.writerow({'Median': median_of_error_list,
                         'IQR': iqr_of_error_list})
        csv_file.close()
        del median_of_error_list
        del iqr_of_error_list
        # Put the results.csv in the belonging folder.
        os.rename(file_name, ''.join([order_path, '/', file_name]))
        # Put the error_normalization_model_#.csv in the belonging folder.
        os.rename(error_normalization_file, ''.join([order_path, '/',
                                                     error_normalization_file]))
    return 'done!'


def main():
    """
    Main.

    This function process the passed arguments using the library 'argparse'.
    Using these arguments it calls the function 'seq_sim' and prints its
    output at the closure.
    """
    # The input arguments for the programm are given either through command
    # line arguments or by their default values
    parser = argparse.ArgumentParser(prog='NN estimate', formatter_class=
                                     argparse.RawDescriptionHelpFormatter,
                                     description='''
    "Deep versus deeper learning in nonparametric regression"
    ______________________________________________________________________
    Process a sequence of simulations for the  neural networks estimate.
    You can either pass any argument using the command line via
    "python3 --input_argument==value(s) ..." or  using default values.
    The input arguments, their types and default values are
    described below.''')
    parser.add_argument('''--n''', metavar='n', type=int, nargs='?',
                        default=200, choices=[100, 200, 500],
                        help='''<class 'int'> - The size of the data sets to
                        sample. The default is 200. Two different data set will
                        be sampled: one for the learning und on for the
                        testing. Their amounts will be n*(4/5) and n/5
                        respectively.''')
    parser.add_argument('''--batch''', nargs='?', metavar='batch',
                        default=True, type=bool, help='''<class 'bool'> -
                        If you wish to perform in batch training, please do
                        not specify this argument. The batch size is hardcoded
                        to n/5. Default value is True''')
    parser.add_argument('''--n_evaluate''', nargs='?', metavar='n_evaluate',
                        default=10000, type=int, help='''<class 'int'> - The
                        size of the data sets for the evaluation phase.
                        The default value is 10000.''')
    parser.add_argument('''--sigma''', nargs='?', metavar='sigma',
                        default=0.05, type=float, help='''<class 'float'> -
                        The size of the data sets for the evaluation phase.
                        The default value is 0.05.''')
    parser.add_argument('''--model''', nargs='?', metavar='model', default=1,
                        type=int, choices=range(1, 6),
                        help='''<class 'int'> - The number of the model to
                        perform simulation(s) for.
                        The default is the first model.''')
    parser.add_argument('''--repititions''', nargs='?', metavar='repititions',
                        default=50, type=int, help='''<class 'int'> - The
                        number of the simulations to perform.
                        The default is 50.''')
    parser.add_argument('''--actfunc''', metavar='actfunc', nargs='?',
                        default='ReLU', type=str,
                        choices=['ReLU', 'sigmoid', 'tanh'],
                        help='''<class 'str'> - The activation function for
                        the neural network(s). "ReLU" is the default.''')
    parser.add_argument('''--initialization''', metavar='initialization',
                        nargs='?', default='small', type=str,
                        choices=['small', 'Schmidt-Hieber', 'general', 'He',
                                 'Xavier'],
                        help='''<class 'str'> - The art to initialize the
                        weight matrices and bias vectors the neural network(s).
                        "small" - regardless the chosen activation
                        function, for "batch" and "no-batch" training.
                        Normal distributed weights with N(0,0.001) and biases
                        as zeros.
                        "Xavier" - for "sigmoid" or "tanh" as an activation
                        function. According to Xavier at al. normal distributed
                        weights and biases as zeros.
                        "He" - for "ReLU" as an activation function.
                        According to He et al. normal distributed weights and
                        biases as zeros.
                        "Schmidt-Hieber" - regardless the chosen activation
                        function. Weights and biases are normal distributed
                        number from [-1,1]''')
    # Save the parsed parameters.
    args = parser.parse_args()
    # Read the parsed parameters for a sequence of simulations.
    n = args.n
    n_learn = int(n*(4/5))
    n_test = int(n/5)
    # Decode the chosen batch size.
    if args.batch:
        batch_size = int(n/5)
    else:
        batch_size = None
    n_evaluate = args.n_evaluate
    sigma = args.sigma
    model = args.model
    repititions = args.repititions
    actfunc = args.actfunc
    initialization = args.initialization
    L_list = [4, 8, 16, 32, 64]
    r_list = [4, 8, 16, 32, 64, 128]
    epochs = 2000
    repmedian = 50

    # Performe a sequence of the simulations with the parsed arguments.
    print(seq_sim(repititions, batch_size, n_learn, n_test, n_evaluate, sigma,
                  model, actfunc, initialization, L_list, r_list,
                  epochs, repmedian))


# main()  # Call the main function.

d, x_learn, y_learn, x_test, y_test, x_evaluate, y_evaluate = generate_data(10000, 2000, 2000, 0.1, 2)
np.savetxt('x_learn.csv',x_learn , delimiter=',')
np.savetxt('y_learn.csv',y_learn , delimiter=',')
np.savetxt('x_test.csv',x_test, delimiter=',')
np.savetxt('y_test.csv',y_test, delimiter=',')
np.savetxt('x_evaluate.csv',x_evaluate , delimiter=',')
np.savetxt('y_evaluate.csv',y_evaluate, delimiter=',')
print(d)
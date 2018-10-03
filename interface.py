from gann import *


def main():
    ################################################
    # Initial dataset choice
    ################################################
    path = 'config/'
    print("Available datasets:")
    for file in sorted(os.listdir(path)):
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
        for key, value in sorted(settings.items()):
            print('{}: {}'.format(key, value))
        print('##########################')

        print('Enter a parameter name to make changes, or continue by pressing enter.. ')
        print('Separate eventual list values with spaces')
        print()
        choice = input('Choice: ')
        if choice in settings.keys():
            new_val = input('New value for ' + choice + ': ')

            # convert list parameters to lists of integers or floats
            if choice in ['network_dimensions', 'initial_weight_range', 'map_layers', 'map_dendrograms', 'display_weights', 'display_biases']:
                new_val = new_val.split(' ')
                new_val = [float(x) if '.' in x else int(x) for x in new_val]
            # check for integers
            elif is_integer(new_val):
                new_val = int(new_val)
            # check for floats
            elif '.' in new_val:
                new_val = float(new_val)

            settings[choice] = new_val
        elif not choice:
            print('Building network ...')
            print()
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
        case_generator = (lambda: TFT.gen_dense_autoencoder_cases(settings['case_count'], settings['data_size'], dr=settings['data_range']))
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

    print(settings['map_dendrograms'])
    print(type(settings['map_dendrograms'][0]))

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
               settings=settings)

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

    return ann


if __name__ == '__main__':
    ann = main()

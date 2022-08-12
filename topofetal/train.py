# - - - - - My imports - - - - - #
import os
import argparse
from utils import DataManager, network_utils

import tensorflow as tf

import pandas as pd

# Fix random
tf.random.set_seed(1312)



def main(p_fold = 0):


    code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    import json
    with open(os.path.join(code_dir, "configs", "config_paths.json")) as jsonFile:
        config_paths = json.load(jsonFile)
        jsonFile.close()

    with open(os.path.join(code_dir, "configs", "config_network.json")) as jsonFile:
        config_network = json.load(jsonFile)
        jsonFile.close()

    feta20_dir = config_paths["feta20_dir"]
    networks_dir = config_paths["networks_dir"]

    participants = pd.read_csv( os.path.join(code_dir, 'configs', 'participants_feta20.csv'))
    participants = participants[ participants['Fold'].isin([0,1,2,3]) ]
    subjects_train = participants[ ~ participants['Fold'].isin([p_fold])]['participant_id']
    subjects_valid = participants[ participants['Fold'].isin([p_fold])]['participant_id']

    paths_t2w_train = [os.path.join(feta20_dir, sub, 'anat', sub+'_rec-irtk_T2w.nii.gz') for sub in subjects_train]
    paths_dseg_train = [os.path.join(feta20_dir, sub, 'anat', sub+'_rec-irtk_dseg.nii.gz') for sub in subjects_train]

    paths_t2w_valid = [os.path.join(feta20_dir, sub, 'anat', sub+'_rec-irtk_T2w.nii.gz') for sub in subjects_valid]
    paths_dseg_valid = [os.path.join(feta20_dir, sub, 'anat', sub+'_rec-irtk_dseg.nii.gz') for sub in subjects_valid]


    dm = DataManager.DataManager( p_patch_size = 64,
                                        p_extraction_step = 48,
                                        p_extraction_axis = [0,1,2],
                                        p_n_classes = 2,
                                        p_n_channels = 1,
                                        p_do_segment_tiv= True)

    ds = tf.data.Dataset.from_tensor_slices((paths_t2w_train, paths_dseg_train)). \
        map(dm.tf_load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
        unbatch(). \
        filter(dm.filter_cortex_patches_ds).\
        map(dm.cast_ds_float32_int16).\
        map(lambda x, y: (tf.image.per_image_standardization(x), y)).\
        cache().\
        map(dm.cast_ds_float32_int16).\
        shuffle(buffer_size=100, seed=1312)

    ds_val = tf.data.Dataset.from_tensor_slices((paths_t2w_valid, paths_dseg_valid)). \
        map(dm.tf_load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE).\
        unbatch(). \
        filter(dm.filter_cortex_patches_ds).\
        map(dm.cast_ds_float32_int16).\
        map(lambda x, y: (tf.image.per_image_standardization(x), y)).\
        cache().\
        shuffle(buffer_size=100, seed=1312)

    p_lb = 0
    p_f = False
    p_mp = config_network["mp"]
    configuration = config_network["configuration"]

    if configuration in ["warm_up", "Baseline", "TopoCP"]:
        lb_hybrid = 0
        if configuration in ["TopoCP"]:
            p_lb = config_network["lambda_topo"]

    elif configuration in ["Hybrid"]:
        lb_hybrid = 0.5



    def gen_network_name_path(config, fold):
        network_name = config + '_' + str(fold)
        c_net_dir = os.path.join(networks_dir, config, network_name)
        network_path = os.path.join(c_net_dir, network_name)
        return network_name, network_path

    network_name, network_path = gen_network_name_path(configuration, p_fold)


    model = network_utils.get_model(dm,
                                    lambda_topoloss     = p_lb,
                                    lambda_hybrid       = lb_hybrid,
                                    min_pers_th         = p_mp)
    model = network_utils.compile_model(p_model = model, p_lr_init = 0.01)

    print("Model compiled.")
    print("-")

    if configuration in ["Baseline", "Hybrid", "TopoCP"]:
        model.load_weights( gen_network_name_path('warm_up', fold)[1] )
        print("Loaded weights from {} ".format( gen_network_name_path('warm_up', fold)[1] ))
        print("-")


    # Tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
     log_dir = network_path + '_tensorboard', histogram_freq=0, write_graph=False,
     write_images=False, update_freq='epoch', profile_batch=2,
     embeddings_freq=0, embeddings_metadata=None)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = network_path + '_min_loss',
        monitor='val_loss',
        mode='min',
        save_weights_only=True,
        save_best_only=True)

    callback_list = []
    callback_list.append(tensorboard_callback)
    callback_list.append(model_checkpoint_callback)

    if configuration in ["warm_up"]:
        init_epoch = 0
        num_epochs = 15
    else:
        init_epoch = 15
        num_epochs = 100

        if configuration in ["TopoCP"]:
            to_be_monitored = 'val_topo'
        else:
            to_be_monitored = 'val_loss'

        earlystopping = tf.keras.callbacks.EarlyStopping(monitor=to_be_monitored,
                                                                   mode="min",
                                                                   min_delta=0.001,
                                                                   patience=6,
                                                                   restore_best_weights=True)
        callback_list.append(earlystopping)


    model.fit(ds
              .map(dm.tf_random_augment_image,
                   num_parallel_calls = tf.data.AUTOTUNE)
              .batch(32),
              epochs=init_epoch+num_epochs,
              validation_data=ds_val.shard(num_shards=3, index=0).batch(32),
              initial_epoch=init_epoch,
              callbacks=callback_list)

    model.save_weights(network_path, save_format='tf')

    return


def get_parser():

    p = argparse.ArgumentParser(description='FetaNet!')
    p.add_argument('--FOLD', type=int, default=0, help='Fold to process')

    return p

if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    fold = int(os.environ['SLURM_ARRAY_TASK_ID']) if 'SLURM_ARRAY_TASK_ID' in os.environ.keys() else args.FOLD

    main(p_fold=fold)

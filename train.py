# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# define model and fit
# save model in result_dir/model/$index, save log in result_dir/log/$index, save plot in result_dir/plot
# train.py temp_dir result_dir params
import os
import sys
import shutil

from keras.models import Sequential, save_model
from keras.layers import Dense, LeakyReLU
from keras.optimizers import adam_v2
import keras.callbacks

import pandas as pd


def main(temp_dir, result_dir, params='params.txt'):
    """
    :param temp_dir:
    :param result_dir:
    :param params:
    :return:

    * Cross-Validation and Estimate
    Divide Dataset into 10 Pieces, two is for test and eight is for train and validation
    Divide Train-Validation Sets into 8 pieces, Seven is for train and One is for Cross-Validation
    Finally, Use Test Sets to Estimate Model
    """
    #
    # Check and Make Directories
    #
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir('result')
    os.mkdir('result/model')
    os.mkdir('result/plot')
    os.mkdir('result/data')

    #
    # Read Hyperparams
    #
    para = []
    if not os.path.exists(params):
        params_f = open(params, 'w')
        params_f.write("FOLD=8")
        params_f.write("EPOCH=256")
        params_f.write("LEARNING_RATE=0.005")
        params_f.write("BATCH=64")
        params_f.close()

    for i, line in enumerate(open(params, 'r')):
        if not i == 2:
            para.append(int(line.strip().split('=')[1]))
        else:
            para.append(float(line.strip().split('=')[1]))

    FOLD, EPOCH, LEARNING_RATE, BATCH = para

    #
    # Define Method to Build Model
    #

    def build_model():
        M = Sequential([
            Dense(5),
            LeakyReLU(),
            Dense(12),
            LeakyReLU(),
            Dense(8),
            LeakyReLU(),
            Dense(1)
        ])
        M.compile(optimizer=adam_v2.Adam(learning_rate=LEARNING_RATE), loss='mse')
        return M

    #
    # Load Data
    #

    dataset = pd.read_csv(os.path.join(temp_dir, 'temp.csv'))
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # Load and Shuffle

    test_set = dataset.sample(frac=0.2).reset_index(drop=True)
    test_set.to_csv(os.path.join(result_dir, 'data', 'test.csv'), index=False)
    train_set = dataset.sample(frac=0.8).reset_index(drop=True)
    test_set.to_csv(os.path.join(result_dir, 'data', 'train.csv'), index=False)

    size = len(train_set)
    size_per_fold = int(size/FOLD)-1
    datasets = []
    for i in range(FOLD):
        datasets.append(train_set.loc[i * size_per_fold:(i + 1) * size_per_fold])     # Divide for C-V

    shutil.rmtree(temp_dir)

    #
    # Fit Model, Save Model
    #

    val_losses = []

    for k, dataset in enumerate(datasets):
        model = build_model()

        train_sets = pd.concat([datasets[i] for i in range(FOLD) if not i == k], axis=0)
        train_ds_y = train_sets.LW_OUT
        train_ds_x = train_sets.drop(['LW_OUT', 'TIMESTAMP', 'SITE'], axis=1)

        val_sets = dataset
        val_ds_y = val_sets.LW_OUT
        val_ds_x = val_sets.drop(['LW_OUT', 'TIMESTAMP', 'SITE'], axis=1)

        CB = keras.callbacks.TensorBoard(log_dir=os.path.join(result_dir, 'logs', str(k)))
        history = model.fit(train_ds_x, train_ds_y, epochs=EPOCH, batch_size=BATCH,
                            validation_data=(val_ds_x, val_ds_y), callbacks=CB)

        export_path = os.path.join(result_dir, 'model', str(k))
        save_model(
            model,
            export_path,
            overwrite=True,
            include_optimizer=True,
            save_format=None,
            signatures=None,
            options=None
        )

        val_loss = model.evaluate(val_ds_x, val_ds_y)
        val_losses.append(val_loss)

    return True


if __name__ == '__main__':
    main(sys.argv[1],
         sys.argv[2])

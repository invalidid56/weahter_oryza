# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# define model and fit
# save model in result_dir/model/$index, save log in result_dir/log/$index, save plot in result_dir/plot
# train.py temp_dir result_dir params
import os
import sys
import shutil

from keras.models import Sequential, save_model
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import adam_v2
import keras.callbacks
import time

import pandas as pd


def main(temp_dir, result_dir, target, params='params.txt'):
    """
    :param temp_dir:
    :param result_dir:
    :param params:
    :param target:
    :return:

    * Cross-Validation and Estimate
    Divide Dataset into 10 Pieces, two is for test and eight is for train and validation
    Divide Train-Validation Sets into 8 pieces, Seven is for train and One is for Cross-Validation
    Finally, Use Test Sets to Estimate Model
    """
    #
    # Check and Make Directories
    #

    data_styles = ['DAY', 'NIGHT']

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    if target == 'RECO':
        for t in ('RECO', 'GPP', 'LEAF'):
            os.mkdir(os.path.join(result_dir, t))
            for data_style in data_styles:
                os.mkdir(os.path.join(result_dir, t, data_style))
                os.mkdir(os.path.join(result_dir, t, data_style, 'model'))
                os.mkdir(os.path.join(result_dir, t, data_style, 'logs'))
                os.mkdir(os.path.join(result_dir, t, data_style, 'plot'))
                os.mkdir(os.path.join(result_dir, t, data_style, 'data'))

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

    def build_model(target):
        if target == 'LEAF':
            M = Sequential([
                Dense(32),
                LeakyReLU(alpha=0.2),
                Dropout(0.5),
                Dense(24),
                LeakyReLU(alpha=0.2),
                Dropout(0.2),
                Dense(16),
                LeakyReLU(alpha=0.2),
                Dropout(0.2),
                Dense(4),
                LeakyReLU(alpha=0.2),
                Dense(1)
            ])

        else:
            M = Sequential([
                Dense(16),
                LeakyReLU(alpha=0.2),
                Dropout(0.2),
                Dense(8),
                LeakyReLU(alpha=0.2),
                Dropout(0.2),
                Dense(4),
                LeakyReLU(alpha=0.2),
                Dense(1)
            ])
        M.compile(optimizer=adam_v2.Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
        return M
    #
    # For Each Styles: Day and Night
    #

    for data_style in data_styles:
        START = time.time()
        print("=====Training at {0} Data=====".format(data_style))

        #
        # Load Data
        #
        dataset = pd.read_csv(os.path.join(temp_dir, target, 'temp_{0}.csv'.format(data_style)))
        dataset = dataset.sample(frac=1).reset_index(drop=True)  # Load and Shuffle

        test_set = dataset.sample(frac=0.2).reset_index(drop=True)
        test_set.to_csv(os.path.join(result_dir, target, data_style, 'data', 'test.csv'), index=False)
        train_set = dataset.sample(frac=0.8).reset_index(drop=True)
        test_set.to_csv(os.path.join(result_dir, target, data_style, 'data', 'train.csv'), index=False)

        size = len(train_set)
        size_per_fold = int(size/FOLD)-1
        datasets = []
        for i in range(FOLD):
            datasets.append(train_set.loc[i * size_per_fold:(i + 1) * size_per_fold])     # Divide for C-V

        #
        # Fit Model, Save Model
        #
        val_losses = []

        for k, dataset in enumerate(datasets):
            model = build_model()

            if target == 'RECO':
                train_sets = pd.concat([datasets[i] for i in range(FOLD) if not i == k], axis=0)
                train_ds_y = train_sets.RECO_DT
                train_ds_x = train_sets.drop(['GPP_DT', 'LEAF', 'RECO_DT',
                                              'TIMESTAMP', 'SITE'], axis=1)

                val_sets = dataset
                val_ds_y = val_sets.RECO_DT
                val_ds_x = val_sets.drop(['GPP_DT', 'LEAF', 'RECO_DT',
                                         'TIMESTAMP', 'SITE'], axis=1)

            elif target == 'LEAF':
                train_sets = pd.concat([datasets[i] for i in range(FOLD) if not i == k], axis=0)
                train_ds_y = train_sets.LEAF
                train_ds_x = train_sets.drop(['LEAF', 'GPP_DT', 'TIMESTAMP', 'SITE'], axis=1)

                val_sets = dataset
                val_ds_y = val_sets.LEAF
                val_ds_x = val_sets.drop(['LEAF', 'GPP_DT', 'TIMESTAMP', 'SITE'], axis=1)

            elif target == 'GPP':
                train_sets = pd.concat([datasets[i] for i in range(FOLD) if not i == k], axis=0)
                train_ds_y = train_sets.GPP_DT
                train_ds_x = train_sets.drop(['GPP_DT', 'TIMESTAMP', 'SITE'], axis=1)

                val_sets = dataset
                val_ds_y = val_sets.GPP_DT
                val_ds_x = val_sets.drop(['GPP_DT', 'TIMESTAMP', 'SITE'], axis=1)

            else:
                print('TARGET VARIABLE ERROR')
                exit()

            CB = keras.callbacks.TensorBoard(log_dir=os.path.join(result_dir, target, data_style, 'logs', str(k)))
            ES = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
            BEST_PATH = os.path.join(result_dir, target, data_style, 'model', str(k), 'best_model.h5')
            MC = keras.callbacks.ModelCheckpoint(filepath=BEST_PATH,
                                                 monitor='val_loss',
                                                 save_best_only=True)

            history = model.fit(train_ds_x, train_ds_y, epochs=EPOCH, batch_size=BATCH,
                                validation_data=(val_ds_x, val_ds_y), callbacks=[CB, ES, MC])

            export_path = os.path.join(result_dir, target, data_style, 'model', str(k))
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

            END = time.time()
            with open(os.path.join(result_dir, target, data_style, 'train_report.txt'), 'w') as report:
                report.write('Time Spent on Training Models : {0}\n'.format(END - START))
                for i, vl in enumerate(val_losses):
                    report.write('Val Loss For Model No. {0} in {1} Style: {2}\n'.format(i, data_style, vl))

    if target == 'GPP_DT':
        shutil.rmtree(temp_dir)

    return True


if __name__ == '__main__':
    main(sys.argv[1],
         sys.argv[2],
         sys.argv[3])

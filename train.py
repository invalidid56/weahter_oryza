# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# define model and fit
# save model in result_dir/model/$index, save log in result_dir/log/$index, save plot in result_dir/plot
# train.py temp_dir result_dir params
import os
import math
import sys
import shutil

from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, LeakyReLU
from keras.optimizers import adam_v2
import keras.callbacks

import pandas as pd
import matplotlib.pyplot as plt


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

    #
    # Read Hyperparams
    #
    para = []
    if not os.path.exists(params):
        params_f = open(params, 'w')
        params_f.write("FOLD = 8")
        params_f.write("LEARNING_RATE = 0.005")
        params_f.write("EPOCH = 4")
        params_f.write("BATCH = 2048")
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
    dataset = dataset.sample(frac=1).reset_index()  # Load and Shuffle

    test_set = dataset.sample(frac=0.01).reset_index()
    train_set = dataset.sample(frac=0.8).reset_index()

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
        train_ds_x = train_sets.drop(['LW_OUT', 'index', 'TIMESTAMP', 'SITE'], axis=1)

        val_sets = dataset
        val_ds_y = val_sets.LW_OUT
        val_ds_x = val_sets.drop(['LW_OUT', 'index', 'TIMESTAMP', 'SITE'], axis=1)

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

    #
    # Estimate, Plot Model
    #

    test_y = test_set.LW_OUT
    test_x = test_set.drop(['LW_OUT', 'index', 'TIMESTAMP', 'SITE'], axis=1)

    # Loss in a Bar

    test_losses = []
    for i in range(FOLD):
        Model = load_model(os.path.join(result_dir, 'model', str(i)))
        eval_result = Model.evaluate(test_x, test_y)
        test_losses.append(math.sqrt(eval_result))

    plt.bar(list(range(FOLD)), test_losses)
    plt.xticks(list(range(FOLD)), list(range(1, FOLD+1)))

    plt.savefig(os.path.join(result_dir, 'plot', 'loss_per_model.png'))

    losses = open(os.path.join(result_dir, 'plot', 'loss.txt'), 'w')
    losses.write(str(test_losses))
    losses.close()

    # CSV: Site, Predict, Real, 1 to 1 plot for a best model
    best_model = test_losses.index(min(test_losses))

    model = load_model(os.path.join(result_dir, 'model', str(best_model)))

    ys_expect = []
    test_x = test_x.values.tolist()[:100]
    for i, x in enumerate(test_x):
        y_expect = model.predict([x]).tolist()[0][0]
        ys_expect.append(y_expect)
    ys_expect = pd.Series(ys_expect)

    result_df = pd.concat([test_set.SITE, test_set.TIMESTAMP, ys_expect, test_y], axis=1)
    result_df.columns = ['SITE', 'TIME', 'EXPECT', 'REAL']
    result_df['LOSS'] = result_df['REAL']-result_df['EXPECT']

    plt.plot(test_x, ys_expect, 'b')
    plt.plot(test_x, test_x)

    plt.savefig(os.path.join(result_dir, 'plot', 'one_to_one.png'))

    result_df.to_csv(os.path.join(result_dir, 'plot', 'result.csv'))


if __name__ == '__main__':
    main(sys.argv[1],
         sys.argv[2])

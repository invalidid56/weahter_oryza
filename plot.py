# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# define model and fit
# save model in result_dir/model/$index, save log in result_dir/log/$index, save plot in result_dir/plot
# train.py temp_dir result_dir params
import os.path
import sys

from keras.models import load_model
import math

import pandas as pd
import matplotlib.pyplot as plt


def main(result_dir, params='params.txt'):
    #
    # Estimate, Plot Model
    #
    para = []
    for i, line in enumerate(open(params, 'r')):
        if not i == 2:
            para.append(int(line.strip().split('=')[1]))
        else:
            para.append(float(line.strip().split('=')[1]))

    FOLD, EPOCH, LEARNING_RATE, BATCH = para

    test_set = pd.read_csv(os.path.join(result_dir, 'data', 'test.csv'))

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

    return True


if __name__ == '__main__':
    main(result_dir=sys.argv[1])
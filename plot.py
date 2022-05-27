# invalidid56@snu.ac.kr 작물생태정보연구실 강준서
# define model and fit
# save model in result_dir/model/$index, save log in result_dir/log/$index, save plot in result_dir/plot
# train.py temp_dir result_dir params
import os.path
import sys
from keras.models import load_model
import math
from sklearn.metrics import r2_score
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

    #
    # For Each Style: Day and Night
    #
    data_styles = ['DAY', 'NIGHT']
    for data_style in data_styles:
        test_set = pd.read_csv(os.path.join(result_dir, data_style, 'data', 'test.csv'))

        test_y = test_set.DIFF_TL
        test_x = test_set.drop(['DIFF_TL', 'TIMESTAMP', 'SITE'], axis=1)

        # Loss in a Bar

        test_losses = []
        for i in range(FOLD):
            Model = load_model(os.path.join(result_dir, data_style, 'model', str(i)))
            eval_result = Model.evaluate(test_x, test_y)
            test_losses.append(math.sqrt(eval_result[1]))   # MAE

        plt.bar(list(range(FOLD)), test_losses)
        plt.xticks(list(range(FOLD)), list(range(1, FOLD+1)))

        plt.savefig(os.path.join(result_dir, data_style, 'plot', 'loss_per_model.png'))
        plt.clf()

        # CSV: Site, Predict, Real, 1 to 1 plot for a best model
        best_model = test_losses.index(min(test_losses))

        model = load_model(os.path.join(result_dir, data_style, 'model', str(best_model)))

        test_x = test_x.values.tolist()
        ys_expect = model.predict(test_x).tolist()
        ys_expect = pd.Series([y[0] for y in ys_expect])

        result_df = pd.concat([test_set.SITE, test_set.TIMESTAMP, ys_expect, test_y], axis=1)
        result_df.columns = ['SITE', 'TIME', 'EXPECT', 'REAL']
        result_df['LOSS'] = result_df['REAL']-result_df['EXPECT']

        plt.plot(test_y, ys_expect, 'bo')
        plt.plot(test_y, test_y, 'r')

        plt.savefig(os.path.join(result_dir, data_style, 'plot', 'one_to_one.png'))
        plt.clf()

        result_df.to_csv(os.path.join(result_dir, data_style, 'plot', 'result.csv'), index=False)

        # Print Estimate Report as a File
        with open(os.path.join('result', data_style, 'estimate_report.txt'), 'w') as report:
            for i, tl in enumerate(test_losses):
                r2 = r2_score(test_y, ys_expect)
                report.write(
                    'Model No. {0} in {1} Style-\n Test Loss: {2}\nR2 Score: {3}\n'.format(i, data_style, tl, r2)
                )

    return True


if __name__ == '__main__':
    main(result_dir=sys.argv[1])

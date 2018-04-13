import numpy
import pandas
from matplotlib import pyplot as plt
from util import plot_answer_prob_curve, fit_linear_stan

column_names=["length_A", "length_B", "abs_difference_percent", "answer", "answer_correct", "duration"]
df = pandas.read_csv("data/rcg_test.csv", names=column_names)

df['difference_A_minus_B'] = df['length_A'] - df['length_B']
df['distance_min'] = df[['length_A', "length_B"]].min(axis=1)
df['difference_relative'] = df['difference_A_minus_B']/df['distance_min']
# print(df.head(3))

n_bins = 20
# ax = plot_answer_prob_curve(df, bins=n_bins); plt.show()

for rel_diff_range_param in [0.10]:
    df_filtered = df[(df['difference_relative'] < rel_diff_range_param).values * (df['difference_relative'] > -rel_diff_range_param).values]
    for frac in [0.01, 0.03, 0.1, 0.3, 1.0]:

        df_to_fit = df_filtered.sample(frac=frac)
        rdiffs = df_to_fit["difference_relative"].values
        answers = df_to_fit["answer"].values

        for n_iter in [1000, 10000]:
            print("-----------------------------------------------")
            print("Sampling using params", rel_diff_range_param, len(rdiffs), frac, n_iter)
            fit = fit_linear_stan(rdiffs, answers, verbose=False)
            x_intercept = fit.extract()['x_intercept']
            print("Results for rel_diff_range " + str(rel_diff_range_param))
            print("Precise values (5th perc., median, 95th perc.) for the x_intercept:", numpy.percentile(x_intercept, [5, 50, 95]))
            print("Error margin", numpy.percentile(x_intercept, 95) - numpy.percentile(x_intercept, 5))
            print("-----------------------------------------------")
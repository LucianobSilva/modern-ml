import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, TextArea
from matplotlib.ticker import FormatStrFormatter

def reduce_mem_usage(df, verbose=False):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if (col_type != object) and (str(col_type) != 'category'):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                int_types = [
                    (np.int8, np.iinfo(np.int8).min, np.iinfo(np.int8).max),
                    (np.uint8, np.iinfo(np.uint8).min, np.iinfo(np.uint8).max),
                    (np.int16, np.iinfo(np.int16).min, np.iinfo(np.int16).max),
                    (np.uint16, np.iinfo(np.uint16).min, np.iinfo(np.uint16).max),
                    (np.int32, np.iinfo(np.int32).min, np.iinfo(np.int32).max),
                    (np.uint32, np.iinfo(np.uint32).min, np.iinfo(np.uint32).max),
                    (np.int64, np.iinfo(np.int64).min, np.iinfo(np.int64).max),
                    (np.uint64, np.iinfo(np.uint64).min, np.iinfo(np.uint64).max)
                ]
                for dtype, min_val, max_val in int_types:
                    if c_min > min_val and c_max < max_val:
                        df[col] = df[col].astype(dtype)
                        if verbose:
                            print(f"Casting column {col} to {str(dtype)}")
                        break
            elif str(col_type)[:5] == 'float':
                float_types = [
                    (np.float16, np.finfo(np.float16).min, np.finfo(np.float16).max),
                    (np.float32, np.finfo(np.float32).min, np.finfo(np.float32).max),
                    (np.float64, np.finfo(np.float64).min, np.finfo(np.float64).max)
                ]
                for dtype, min_val, max_val in float_types:
                    if c_min > min_val and c_max < max_val:
                        df[col] = df[col].astype(dtype)
                        if verbose:
                            print(f"Casting column {col} to {str(dtype)}")
                        break

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def clf_metric_report(y_score, y_true):
    print('Evaluating the model...')
    roc_auc = roc_auc_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    logloss = log_loss(y_true, y_score)
    print(f'ROC AUC: {roc_auc}')
    print(f'Brier Score: {brier}')
    print(f'Average Precision: {avg_precision}')
    print(f'Log Loss: {logloss}')

def compute_and_plot_permutation_importance(model, X_test, y_test, metric='average_precision', n_repeats=5):
    # Calculate permutation importance
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=42, scoring=metric)
    features = X_test.columns.to_list()

    # Sort features by importance
    feature_importance = pd.DataFrame({'feature': features, 'importance': result.importances_mean})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # Plot top 20 most important features using seaborn
    plt.figure(figsize=(10, 12))
    sns.barplot(data=feature_importance, y='feature', x='importance')
    plt.xlabel('Permutation Importance')
    plt.ylabel('Features')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.show()
    return feature_importance

def plot_calibration_curve(y_score, y_true, title='Calibration Curve'):
    prob_true, prob_pred = calibration_curve(y_score, y_true, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title(title)
    plt.show()

def plot_pr_calib_curve(y_score, y_true):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')

    plt.subplot(1, 2, 2)
    plt.plot(prob_pred, prob_true, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')

    plt.tight_layout()
    plt.show()

def plot_dis_probs(y_score, y_true):
    plt.figure(figsize=(10, 6))
    sns.histplot(y_score[y_true == 1], bins=50, color='red', label='Churn', kde=True, stat='density')
    sns.histplot(y_score[y_true == 0], bins=50, color='blue', label='Non-Churn', kde=True, stat='density')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities for Churn vs Non-Churn')
    plt.legend()
    plt.show()


def plot_prediction_intervals(
    title,
    axs,
    y_test_sorted,
    y_pred_sorted,
    lower_bound,
    upper_bound,
    coverage,
    width,
    num_plots_idx
):
    """
    Plot of the prediction intervals for each different conformal
    method.
    """
    round_to = 3
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.0f' + "k"))
    axs.xaxis.set_major_formatter(FormatStrFormatter('%.0f' + "k"))

    lower_bound_ = np.take(lower_bound, num_plots_idx)
    y_pred_sorted_ = np.take(y_pred_sorted, num_plots_idx)
    y_test_sorted_ = np.take(y_test_sorted, num_plots_idx)

    error = y_pred_sorted_-lower_bound_

    warning1 = y_test_sorted_ > y_pred_sorted_+error
    warning2 = y_test_sorted_ < y_pred_sorted_-error
    warnings = warning1 + warning2
    axs.errorbar(
        y_test_sorted_[~warnings],
        y_pred_sorted_[~warnings],
        yerr=np.abs(error[~warnings]),
        capsize=5, marker="o", elinewidth=2, linewidth=0,
        label="Inside prediction interval"
        )
    axs.errorbar(
        y_test_sorted_[warnings],
        y_pred_sorted_[warnings],
        yerr=np.abs(error[warnings]),
        capsize=5, marker="o", elinewidth=2, linewidth=0, color="red",
        label="Outside prediction interval"
        )
    axs.scatter(
        y_test_sorted_[warnings],
        y_test_sorted_[warnings],
        marker="*", color="green",
        label="True value"
    )
    axs.set_xlabel("True house prices in $")
    axs.set_ylabel("Prediction of house prices in $")
    ab = AnnotationBbox(
        TextArea(
            f"Coverage: {np.round(coverage, round_to)}\n"
            + f"Interval width: {np.round(width, round_to)}"
        ),
        xy=(np.min(y_test_sorted_)*3, np.max(y_pred_sorted_+error)*0.95),
        )
    lims = [
        np.min([axs.get_xlim(), axs.get_ylim()]),  # min of both axes
        np.max([axs.get_xlim(), axs.get_ylim()]),  # max of both axes
    ]
    axs.plot(lims, lims, '--', alpha=0.75, color="black", label="x=y")
    axs.add_artist(ab)
    axs.set_title(title, fontweight='bold')

def plot_shap_values(shap_values, X_test, figsize=(10, 12), type='bar'):
    #plt.figure(figsize=figsize)
    shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=25)
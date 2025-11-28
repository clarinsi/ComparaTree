from scipy.stats import shapiro

def check_for_normality(data, sign_level=0.05):
    shapiro_test = shapiro(data)

    return (shapiro_test.pvalue < sign_level, shapiro_test.pvalue, shapiro_test.statistic)

import sys
import pandas as pd
from sklearn import ensemble, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import warnings
warnings.filterwarnings('ignore')

def mape(y_true, y_pred):
    """Return the Mean Absolute Percentage Error
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    """
    return pd.Series(((y_pred - y_true) / y_true).abs() * 100).mean()

available_systems = ["Apache", "BerkeleyC", "BerkeleyJ", "Dune", "HIPAcc", "HSMGP", "LLVM", "SQLite", "Linux"]


if len(sys.argv) < 2:
    sys.exit("The script needs a system name as parameter")
    
system_name = sys.argv[1]

print("Running experiments for {}".format(system_name))


def get_dataset(name):
    if name == "Linux":
        size_columns = ["GZIP-bzImage", "GZIP-vmlinux", "GZIP", "BZIP2-bzImage", "vmlinux", 
              "BZIP2-vmlinux", "BZIP2", "LZMA-bzImage", "LZMA-vmlinux", "LZMA", "XZ-bzImage", "XZ-vmlinux", "XZ", 
              "LZO-bzImage", "LZO-vmlinux", "LZO", "LZ4-bzImage", "LZ4-vmlinux", "LZ4"]
        df = pd.read_pickle("datasets/dataset_413.pkl")
        df["perf"] = df["vmlinux"]
        return df.drop(columns=size_columns)
    else :
        return pd.read_csv("datasets/{}.csv".format(name))
    
df = get_dataset(system_name)

# Experimental parameters
## Thresholds, or performance objectives, are defined according to the dataset performance distribution
thresholds = [df["perf"].quantile(i) for i in [0.1,0.2,0.5,0.8,0.9]]
## Size of the training set, up to 70% to keep at least 30% of the dataset as testing set
training_sizes = [0.1,0.2,0.5,0.7]
## Number of times each training is repeated. You might want to reduce it to 5 for Linux as it would be very long and not that useful as it is quite stable
n_repeats = 20
## Parameter only for spcialized regression, the gap to be created in the performance distribution at the threshold level
gaps = [df["perf"].max(), df["perf"].mean(), df["perf"].mean()/2, df["perf"].mean()/4]

# Hypermarameters grid search values
classification_criterion = ["gini","entropy"]
regression_criterion = ["mse","friedman_mse"]

max_depths = [5,8,10,12,14,16,18,20]
min_samples_splits = [2,5,10,20,50,100]


def train_classification_tree(hp, threshold, train_size):
    accuracy = []
    # Repeat the learning a set number of time
    for _ in range(n_repeats):
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["perf"]), df["perf"] > threshold, train_size=train_size)
        
        # Set Decision Tree hyperparameters
        clf = tree.DecisionTreeClassifier(**hp)
        # Train it
        clf.fit(X_train, y_train)
        
        # Save the balanced accuracy
        accuracy.append(balanced_accuracy_score(
            y_test,
            clf.predict(X_test)
        ))
    acc = pd.Series(accuracy)
    # Report the average accuracy and the standard deviation
    return acc.mean(), acc.std()

classification = []

# For each combination of parameters and hyperparameters, perform the experiment
for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        for criterion in classification_criterion:
            for training_size in training_sizes:
                for threshold in thresholds:
                    hp = {
                        "max_depth":max_depth,
                        "min_samples_split":min_samples_split,
                        "criterion":criterion
                    }
                    mean, std = train_classification_tree(hp, threshold, training_size)
                    hp["threshold"] = threshold
                    hp["training_size"] = training_size
                    hp["mean"] = mean
                    hp["std"] = std
                    classification.append(hp)
                    
# Import data into a DataFrame
df_classification = pd.DataFrame(classification)

def train_regression_tree(hp, thresholds, train_size):
    accuracy = {t:[] for t in thresholds}
    for _ in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["perf"]), df["perf"], train_size=train_size)
        
        clf = tree.DecisionTreeRegressor(**hp)
        clf.fit(X_train, y_train)
        
        # Save the balanced accuracy for each threshold, as the regression model is threshold agnostic
        for threshold in thresholds:
            accuracy[threshold].append(balanced_accuracy_score(
                y_test > threshold,
                clf.predict(X_test) > threshold
            ))
    # Report average balanced accuracy and the stadard deviation indexed by threshold
    return {t:{"mean":pd.Series(accuracy[threshold]).mean(),"std":pd.Series(accuracy[threshold]).std()} for t in thresholds}

regression = []

for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        for criterion in regression_criterion:
            for training_size in training_sizes:
                hp = {
                    "max_depth":max_depth,
                    "min_samples_split":min_samples_split,
                    "criterion":criterion
                }
                for threshold, i in train_regression_tree(hp, thresholds, training_size).items():

                    i["max_depth"] = max_depth
                    i["min_samples_split"] = min_samples_split
                    i["criterion"] = criterion
                    i["threshold"] = threshold
                    i["training_size"] = training_size
                    regression.append(i)
                    
df_regression = pd.DataFrame(regression)


def train_specialized_regression_tree(hp, threshold, train_size, gap):
    accuracy = []
    for _ in range(n_repeats):
        # .map(lambda x : x + gap if x > threshold else x) <- this creates the gap
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["perf"]), df["perf"].map(lambda x : x + gap if x > threshold else x), train_size=train_size)
        
        clf = tree.DecisionTreeRegressor(**hp)
        clf.fit(X_train, y_train)
        accuracy.append(balanced_accuracy_score(
            y_test > threshold,
            clf.predict(X_test) > threshold
        ))
    acc = pd.Series(accuracy)
    return acc.mean(), acc.std()

specialized_regression = []

for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        for criterion in regression_criterion:
            for training_size in training_sizes:
                for threshold in thresholds:
                    for gap in gaps:
                        hp = {
                            "max_depth":max_depth,
                            "min_samples_split":min_samples_split,
                            "criterion":criterion
                        }
                        mean, std = train_specialized_regression_tree(hp, threshold, training_size, gap)
                        hp["threshold"] = threshold
                        hp["training_size"] = training_size
                        hp["gap"] = gap
                        hp["mean"] = mean
                        hp["std"] = std
                        specialized_regression.append(hp)
                        
df_spec_regression = pd.DataFrame(specialized_regression)


frl_models = []
for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["perf"]), df["perf"], train_size=0.1)
    reg = ensemble.RandomForestRegressor(max_depth=12, min_samples_split=5, criterion="mse", n_jobs=-1)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    acc = mape(
        y_test,
        reg.predict(X_test)
    )

    frl_models.append({
        "model":reg,
        "error":acc
    })
    
df_importance = pd.DataFrame([i["model"].feature_importances_ for i in frl_models], columns=X_train.columns)
df_importance.loc["mean"] = df_importance.mean()

df_values = df_importance.T
for i in df_values.columns:
    df_values["ranking-"+str(i)] = df_values[i].sort_values(ascending=False).rank(method="min", ascending=False)
    
feature_ranking_list = list(df_values.sort_values("ranking-mean")["ranking-mean"].index)


# Number of options to consider
list_n_options = list(range(1,len(feature_ranking_list)))
# Linux has too many options to consider all
if system_name == "Linux":
    list_n_options = [150,200,250,300,350,400,450,500,1000]
    
def train_classification_tree_fs(hp, threshold, train_size, features):
    accuracy = []
    for _ in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(df[features], df["perf"] > threshold, train_size=train_size)
        
        clf = tree.DecisionTreeClassifier(**hp)
        clf.fit(X_train, y_train)

        accuracy.append(balanced_accuracy_score(
            y_test,
            clf.predict(X_test)
        ))
    acc = pd.Series(accuracy)
    return acc.mean(), acc.std()

classification_fs = []

for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        for criterion in classification_criterion:
            for training_size in training_sizes:
                for threshold in thresholds:
                    for n_options in list_n_options:
                        hp = {
                            "max_depth":max_depth,
                            "min_samples_split":min_samples_split,
                            "criterion":criterion
                        }
                        mean, std = train_classification_tree_fs(hp, threshold, training_size, feature_ranking_list[:n_options])
                        hp["threshold"] = threshold
                        hp["training_size"] = training_size
                        hp["n_options"] = n_options
                        hp["mean"] = mean
                        hp["std"] = std
                        classification_fs.append(hp)
                        
df_classification_fs = pd.DataFrame(classification_fs)


def train_regression_tree_fs(hp, thresholds, train_size, features):
    accuracy = {t:[] for t in thresholds}
    for _ in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(df[features], df["perf"], train_size=train_size)
        
        clf = tree.DecisionTreeRegressor(**hp)
        clf.fit(X_train, y_train)
        
        for threshold in thresholds:
            accuracy[threshold].append(balanced_accuracy_score(
                y_test > threshold,
                clf.predict(X_test) > threshold
            ))
    return {t:{"mean":pd.Series(accuracy[threshold]).mean(),"std":pd.Series(accuracy[threshold]).std()} for t in thresholds}

regression_fs = []

for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        for criterion in regression_criterion:
            for training_size in training_sizes:
                for n_options in list_n_options:
                    hp = {
                        "max_depth":max_depth,
                        "min_samples_split":min_samples_split,
                        "criterion":criterion
                    }
                    for threshold, i in train_regression_tree_fs(hp, thresholds, training_size, feature_ranking_list[:n_options]).items():

                        i["max_depth"] = max_depth
                        i["min_samples_split"] = min_samples_split
                        i["criterion"] = criterion
                        i["threshold"] = threshold
                        i["training_size"] = training_size
                        i["n_options"] = n_options
                        regression_fs.append(i)
                        
df_regression_fs = pd.DataFrame(regression_fs)

def train_specialized_regression_tree_fs(hp, threshold, train_size, gap, features):
    accuracy = []
    for _ in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(df[features], df["perf"].map(lambda x : x + gap if x > threshold else x), train_size=train_size)
        
        clf = tree.DecisionTreeRegressor(**hp)
        clf.fit(X_train, y_train)
        accuracy.append(balanced_accuracy_score(
            y_test > threshold,
            clf.predict(X_test) > threshold
        ))
    acc = pd.Series(accuracy)
    return acc.mean(), acc.std()

specialized_regression_fs = []

for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        for criterion in regression_criterion:
            for training_size in training_sizes:
                for threshold in thresholds:
                    for gap in gaps:
                        for n_options in list_n_options:
                            hp = {
                                "max_depth":max_depth,
                                "min_samples_split":min_samples_split,
                                "criterion":criterion
                            }
                            mean, std = train_specialized_regression_tree_fs(hp, threshold, training_size, gap, feature_ranking_list[:n_options])
                            hp["threshold"] = threshold
                            hp["training_size"] = training_size
                            hp["gap"] = gap
                            hp["n_options"] = n_options
                            hp["mean"] = mean
                            hp["std"] = std
                            specialized_regression_fs.append(hp)
                            
                            
df_spec_regression_fs = pd.DataFrame(specialized_regression_fs)


df_all = pd.concat([df_classification, df_classification_fs, df_regression, df_regression_fs, df_spec_regression, df_spec_regression_fs])

s = system_name
s += " & "
s += "{:.1f}\\%".format(df_all.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].mean() * 100)
s += r" ($\pm${:.1f})".format(df_all.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].std() * 100)
s += " & "
s += "{:.1f}\\%".format(df_classification.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].mean() * 100)
s += r" ($\pm${:.1f})".format(df_classification.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].std() * 100)
s += " & "
s += "{:.1f}\\%".format(df_classification_fs.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].mean() * 100)
s += r" ($\pm${:.1f})".format(df_classification_fs.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].std() * 100)
s += " & "
s += "{:.1f}".format(df_regression.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].mean() * 100)
s += r" ($\pm${:.1f})".format(df_regression.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].std() * 100)
s += " & "
s += "{:.1f}\\%".format(df_regression_fs.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].mean() * 100)
s += r" ($\pm${:.1f})".format(df_regression_fs.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].std() * 100)
s += " & "
s += "{:.1f}\\%".format(df_spec_regression.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].mean() * 100)
s += r" ($\pm${:.1f})".format(df_spec_regression.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].std() * 100)
s += " & "
s += "{:.1f}\\%".format(df_spec_regression_fs.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].mean() * 100)
s += r" ($\pm${:.1f})".format(df_spec_regression_fs.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[0.7].std() * 100)

print(s)

text_lines = []

text_lines.append("\\begin{table*}")
text_lines.append("\\begin{tabular}{ |l|ccccc| }")
text_lines.append("\\hline")
text_lines.append("\\multirow{2}{*}{Training set size} & \\multicolumn{4}{c}{\\hspace{2cm}Acceptable configurations} & \\\\")

text_lines.append("  &  10\% & 20\% & 50\% & 80\% & 90\% \\\\")

text_lines.append("\\hline \\hline")
text_lines.append("&\\multicolumn{5}{c|}{\\textbf{Classification}}&")
text_lines.append("\\hline")

for k,i in df_classification.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').iterrows():
    s = "{:.0f}".format(int(k* df.shape[0]))
    for l,j in i.iteritems():
        j_fs = df_classification_fs.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[k,l]
        diff = j_fs - j
        j_best = j_fs if j_fs > j else j
        best = j_best == df_all.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[k,l]
        color = "\\textcolor{ForestGreen}{+" if diff > 0 else "\\textcolor{red}{-"
        if abs(diff) <= 0.01:
            color = color.replace("red","gray").replace("ForestGreen","gray")
        s += " & {}{:.1f}{} ({}{:0.1f}{})".format(
            "\\textbf{" if best else "", 
            j_best*100, 
            "}" if best else "", 
            color,
            abs(diff*100),
            "}"
        )
    s += " \\\\"
    text_lines.append(s)
    
text_lines.append("\\hline \\hline")
text_lines.append("&\\multicolumn{5}{c|}{\\textbf{Regression}}&")
text_lines.append("\\hline")
for k,i in df_regression.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').iterrows():
    s = "{:.0f}".format(int(k* df.shape[0]))
    for l,j in i.iteritems():
        j_fs = df_regression_fs.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[k,l]
        diff = j_fs - j
        j_best = j_fs if j_fs > j else j
        best = j_best == df_all.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[k,l]
        color = "\\textcolor{ForestGreen}{+" if diff > 0 else "\\textcolor{red}{-"
        color = color if diff >= 0.01 or diff <= -0.01 else "\\textcolor{gray}{+"
        
        s += " & {}{:.1f}{} ({}{:0.1f}{})".format(
            "\\textbf{" if best else "", 
            j_best*100, 
            "}" if best else "", 
            color,
            abs(diff*100),
            "}"
        )
    s += " \\\\"
    text_lines.append(s)

text_lines.append("\\hline \\hline")
text_lines.append("&\\multicolumn{5}{c|}{\\textbf{Specialized Regression}}&")
text_lines.append("\\hline")
for k,i in df_spec_regression.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').iterrows():
    s = "{:.0f}".format(int(k* df.shape[0]))
    for l,j in i.iteritems():
        j_fs = df_spec_regression_fs.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[k,l]
        diff = j_fs - j
        j_best = j_fs if j_fs > j else j
        best = j_best == df_all.groupby(["threshold","training_size"])["mean"].max().unstack('threshold').loc[k,l]
        color = "\\textcolor{ForestGreen}{+" if diff > 0 else "\\textcolor{red}{-"
        color = color if diff >= 0.01 or diff <= -0.01 else "\\textcolor{gray}{+"
        s += " & {}{:.1f}{} ({}{:0.1f}{})".format(
            "\\textbf{" if best else "", 
            j_best*100, 
            "}" if best else "", 
            color,
            abs(diff*100),
            "}"
        )
    s += " \\\\"
    text_lines.append(s)
    
text_lines.append("\\hline")
text_lines.append("\\end{tabular}")
text_lines.append("\\caption{Decision tree classification accuracy on performance specialization for %s on three strategies. Bold represents the best result among other strategies including feature selection, the value in brackets is the difference made by feature selection\\label{tab:%s}}" % (system_name,system_name.lower()))
text_lines.append("\\end{table*}")

text_lines.append("\\begin{table*}")

with open("{}_table.tex".format(system_name),"w") as f:
    f.write("\n".join(text_lines))
    
    
import time
threshold = df["perf"].median()
train_size = 0.7

# Getting the best hyperparameters
best_config_classification = df_classification.query("threshold == {} and training_size == {}".format(threshold,train_size)).sort_values("mean", ascending=False).iloc[0]
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["perf"]), df["perf"] > threshold, train_size=train_size)
clf = tree.DecisionTreeClassifier(max_depth=best_config_classification["max_depth"], min_samples_split=best_config_classification["min_samples_split"], criterion=best_config_classification["criterion"])

# Start timing
time_begin = time.time()

# Run the learning process
for _ in range(0,10):
    clf.fit(X_train, y_train)

time_classification = (time.time() - time_begin) / 10



best_config_classification_fs = df_classification_fs.query("threshold == {} and training_size == {}".format(threshold,train_size)).sort_values("mean", ascending=False).iloc[0]
X_train, X_test, y_train, y_test = train_test_split(df[feature_ranking_list[:best_config_classification_fs["n_options"]]], df["perf"] > threshold, train_size=train_size)
clf = tree.DecisionTreeClassifier(max_depth=best_config_classification_fs["max_depth"], min_samples_split=best_config_classification_fs["min_samples_split"], criterion=best_config_classification_fs["criterion"])

time_begin = time.time()

for _ in range(0,10):
    clf.fit(X_train, y_train)

time_classification_fs = (time.time() - time_begin) / 10




best_config_regression = df_regression.query("threshold == {} and training_size == {}".format(threshold,train_size)).sort_values("mean", ascending=False).iloc[0]
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["perf"]), df["perf"], train_size=train_size)
clf = tree.DecisionTreeRegressor(max_depth=best_config_regression["max_depth"], min_samples_split=best_config_regression["min_samples_split"], criterion=best_config_regression["criterion"])

time_begin = time.time()

for _ in range(0,10):
    clf.fit(X_train, y_train)

time_regression = (time.time() - time_begin) / 10



best_config_regression_fs = df_regression_fs.query("threshold == {} and training_size == {}".format(threshold,train_size)).sort_values("mean", ascending=False).iloc[0]
X_train, X_test, y_train, y_test = train_test_split(df[feature_ranking_list[:best_config_regression_fs["n_options"]]], df["perf"], train_size=train_size)
clf = tree.DecisionTreeRegressor(max_depth=best_config_regression_fs["max_depth"], min_samples_split=best_config_regression_fs["min_samples_split"], criterion=best_config_regression_fs["criterion"])

time_begin = time.time()

for _ in range(0,10):
    clf.fit(X_train, y_train)

time_regression_fs = (time.time() - time_begin) / 10



best_config_spec_regression = df_spec_regression.query("threshold == {} and training_size == {}".format(threshold,train_size)).sort_values("mean", ascending=False).iloc[0]
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["perf"]), df["perf"].map(lambda x : x + best_config_spec_regression["gap"] if x > threshold else x), train_size=train_size)
clf = tree.DecisionTreeRegressor(max_depth=best_config_spec_regression["max_depth"], min_samples_split=best_config_spec_regression["min_samples_split"], criterion=best_config_spec_regression["criterion"])

time_begin = time.time()

for _ in range(0,10):
    clf.fit(X_train, y_train)

time_spec_regression = (time.time() - time_begin) / 10



best_config_spec_regression_fs = df_spec_regression_fs.query("threshold == {} and training_size == {}".format(threshold,train_size)).sort_values("mean", ascending=False).iloc[0]
X_train, X_test, y_train, y_test = train_test_split(df[feature_ranking_list[:best_config_spec_regression_fs["n_options"]]], df["perf"].map(lambda x : x + best_config_spec_regression_fs["gap"] if x > threshold else x), train_size=train_size)
clf = tree.DecisionTreeRegressor(max_depth=best_config_spec_regression["max_depth"], min_samples_split=best_config_spec_regression["min_samples_split"], criterion=best_config_spec_regression["criterion"])

time_begin = time.time()

for _ in range(0,10):
    clf.fit(X_train, y_train)

time_spec_regression_fs = (time.time() - time_begin) / 10

print("Training time classification : ", time_classification)
print("Training time classification FS : ", time_classification_fs)

print("Training time regression : ", time_regression)
print("Training time regression FS : ", time_regression_fs)

print("Training time spec regression : ", time_spec_regression)
print("Training time spec regression FS : ", time_spec_regression_fs)
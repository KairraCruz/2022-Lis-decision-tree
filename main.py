import graphviz
import pandas as pd
from pprint import pprint
from sklearn import tree, preprocessing
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

column_consent = "PRIVACY CONSENT. I understand and agree that by filling out this form, I am allowing the researcher (Kairra Cruz) to collect, process, use, share, and disclose my personal information and also to store it as long as necessary for the fulfillment of Facilities Management Capstone Survey of the stated purpose and in accordance with applicable laws, including the Data Privacy Act of 2012 and its Implementing Rules and Regulations. The purpose and extent of the collection, use, sharing, disclosure, and storage of my personal information were cleared to me. "
column_multiple_choices = '6. What kind of service/s do you usually request? You can choose more than 1.'
column_suggestions = "Suggestions to improve existing facilities management processes."
columns_irrelevant_dataset = [column_consent, column_suggestions, "Timestamp", "Email Address"]
columns_multiple_choice = [column_multiple_choices]
columns_exclude_one_hot = []

ranking_order = ["Very Dissatisfied", "Dissatisfied", "Satisfied", "Very Satisfied"]
    
target_output = '26. The overall quality of the work'

output_suggestions = "suggestions.txt"
output_pngfile = "report.png"
output_textfile = "report.txt"


def load_csv():
    filename = "responses.csv"

    return pd.read_csv(filename)

def filter_consent(records):
    # filter out responses that did not answer no

    records_consent_no = records[records[column_consent] != "Yes"]
    print(f"{len(records_consent_no)} in record replied with 'no consent'")

    return records[records[column_consent] == "Yes"]

def cleanup_suggestions(records):
    # filter out suggestions
    records_suggestions = records[
        records[column_suggestions].notna()
    ]

    suggestions = records_suggestions[column_suggestions].values.tolist()

    # clean up suggestions
    suggestions = [
        suggestion
        for suggestion in suggestions
        if suggestion and suggestion.strip()
    ]
    suggestions = [
        suggestion
        for suggestion in suggestions
        if (
            "none" not in suggestion.lower() and
            "nothing" not in suggestion.lower() and
            "no comment" not in suggestion.lower()
           )
    ]

    suggestions.sort()
    
    with open(output_suggestions, "w") as f:
        for line in suggestions:
            f.write(line)
            f.write("\n")

    print(f"{len(suggestions)} suggestions found. Saved to {output_suggestions}")

    return records, suggestions

def filter_irrelevant_columns(records):
    # filter out irrelevant columns
    return records.drop(columns_irrelevant_dataset, axis=1)

def get_target_output(records):
    # Ask the user what the target output is
    default_target_output = -1

    for i, column in enumerate(records.columns):
        if column == target_output:
            print("(default) ", end="")
            default_target_output = i + 1
        
        print(f"'{column}'")

    choice = input("Choose column to be output (leave empty for default): ").strip()
    if choice == "":
        choice = default_target_output
    chosen = records.columns[int(choice) - 1]

    return chosen

# https://stackoverflow.com/a/52935270/1599
def one_hot_encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)

    return res

# Transform multiple choices to list
# Then one-hot encode records
# https://stackoverflow.com/a/45312840/1599
def split_and_one_hot_encode(records, columns_multiple_choice):
    def split(x):
        # Split the cell, but prevent splitting those that have , in their responses
        # 1. Replace "Top," with "Top|"
        # 2. Replace "Heating, Ventilation, and" with "Heating| Ventilation| and"
        # 3. split the cell with ,
        # 4. Reverse step 1 and 2

        x = x.replace("Top,", "Top|")
        x = x.replace("Heating, Ventilation, and", "Heating| Ventilation| and")
        x = x.split(",")
        x = [_.replace("Top|", "Top,") for _ in x]
        x = [_.replace("Heating| Ventilation| and", "Heating, Ventilation, and") for _ in x]
        x = [_.strip() for _ in x]
        x = [_ for _ in x if _]
        return x
        
    for column in columns_multiple_choice:
        records[column_multiple_choices] = records[column].apply(split)

        mlb = MultiLabelBinarizer(sparse_output=True)

        records = records.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(
                    records.pop(column)),
                    index=records.index,
                    columns=mlb.classes_
                )
        )

    return records

def one_hot_encode(records, columns_exclude_one_hot):
    for column in records.columns:
        if column == target_output or column in columns_exclude_one_hot:
            continue

        records = one_hot_encode_and_bind(records, column)

    return records

def label_encode(records, columns_exclude_one_hot):    
    # https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
    le = LabelEncoder()
    le.fit(ranking_order)

    for column in columns_exclude_one_hot:
        records[column] = le.transform(records[column])

    return records

def preprocess_records_for_fitting(records, target_output):
    # preprocess records for fitting to decision tree
    # 1. One hot encode responses with multiple choices
    # 2. Label encode
    # 3. One hot encode responses everything else

    records = split_and_one_hot_encode(records, columns_multiple_choice)
    records = label_encode(records, columns_exclude_one_hot)
    records = one_hot_encode(records, columns_exclude_one_hot)

    return records

def fit_to_model(records, target_output):
    # generate decision tree    
    X = records.loc[:, records.columns != target_output]
    y = records[target_output]
    classifier = tree.DecisionTreeClassifier().fit(X, y)

    print(f"Target column is: {target_output}")
    print(f"Outputs are: {y.unique()}")

    return classifier, X, y

def generate_output(classifier, X, y):
    def generate_png():
        # Graphical tree:
        # https://scikit-learn.org/stable/modules/tree.html#classification
        # https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html#sklearn.tree.export_graphviz
        #tree.plot_tree(classifier)
        dot_data = tree.export_graphviz(classifier,
                                        feature_names=X.columns.tolist(),  
                                        class_names= y.unique(), 
                                        filled=True, rounded=True,  
                                        special_characters=True,
                                        impurity=False,)
        graph = graphviz.Source(dot_data, format='png')
        graph.render(output_pngfile.replace(".png", "")) 
        print(f"Output written to {output_pngfile}")
    
    def generate_txt():
        # Text tree
        report = tree.export_text(classifier,
                                  feature_names=X.columns.tolist())
    
        #for i, class_name in enumerate(output_label.classes_):
        #    export = export.replace(f"class: {i}", f"class: {class_name}")
        #report = report.replace(f"<= 0.50", "== FALSE")
        #report = report.replace(f">  0.50", "== TRUE")
        
        with open(output_textfile, "w") as f:
            f.write(report)
        
        print(f"Output written to {output_textfile}")

    generate_png()
    generate_txt()

def main():
    records = load_csv()
    records = filter_consent(records)
    suggestions = cleanup_suggestions(records)
    records = filter_irrelevant_columns(records)
    target_output = get_target_output(records)

    records = preprocess_records_for_fitting(records, target_output)
    classifier, X, y = fit_to_model(records, target_output)

    generate_output(classifier, X, y)


if __name__ == "__main__":
    main()
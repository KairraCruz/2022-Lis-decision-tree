import graphviz
import pandas as pd
from pprint import pprint
from sklearn import tree, preprocessing
        
header_consent = "PRIVACY CONSENT. I understand and agree that by filling out this form, I am allowing the researcher (Kairra Cruz) to collect, process, use, share, and disclose my personal information and also to store it as long as necessary for the fulfillment of Facilities Management Capstone Survey of the stated purpose and in accordance with applicable laws, including the Data Privacy Act of 2012 and its Implementing Rules and Regulations. The purpose and extent of the collection, use, sharing, disclosure, and storage of my personal information were cleared to me. "
header_multiple_choices = '6. What kind of service/s do you usually request? You can choose more than 1.'
header_suggestions = "Suggestions to improve existing facilities management processes."
headers_irrelevant_dataset = [header_consent, header_suggestions, "Timestamp", "Email Address"]
headers_multiple_choice = [header_multiple_choices]

target_output = '26. The overall quality of the work'

output_pngfile = "report.png"
output_textfile = "report.txt"


def load_csv():
    filename = "responses.csv"

    return pd.read_csv(filename)

def filter_consent(records):
    # filter out responses that did not answer no

    records_consent_no = records[records[header_consent] != "Yes"]
    print(f"{len(records_consent_no)} in record replied with 'no consent'")

    return records[records[header_consent] == "Yes"]

def cleanup_suggestions(records):
    # filter out suggestions
    records_suggestions = records[
        records[header_suggestions].notna()
    ]

    suggestions = records_suggestions[header_suggestions].values.tolist()

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
    pprint(suggestions)
    print(f"{len(suggestions)} suggestions found")

    return records, suggestions

def filter_irrelevant_columns(records):
    # filter out irrelevant columns
    return records.drop(headers_irrelevant_dataset, axis=1)

# https://stackoverflow.com/a/52935270/1599
def one_hot_encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)

    return res

# Transform multiple choices to list
# Then one-hot encode records
# https://stackoverflow.com/a/45312840/1599
def split_and_one_hot_encode(records, headers_multiple_choice):
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
        
    for header in headers_multiple_choice:
        records[header_multiple_choices] = records[header].apply(split)
        from sklearn.preprocessing import MultiLabelBinarizer

        mlb = MultiLabelBinarizer(sparse_output=True)

        records = records.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(
                    records.pop(header)),
                    index=records.index,
                    columns=mlb.classes_
                )
        )

    return records

def preprocess_records_for_fitting(records):
    # preprocess records for fitting to decision tree
    # 1. One hot encode responses with multiple choices
    # 2. One hot encode responses everything else
    
    records = split_and_one_hot_encode(records, headers_multiple_choice)
    
    for header in records.columns:
        if header == target_output:
            continue

        records = one_hot_encode_and_bind(records, header)

    return records

def fit_to_model(records, target_output):
    # generate decision tree    
    X = records.loc[:, records.columns != target_output]
    y = records[target_output]
    classifier = tree.DecisionTreeClassifier().fit(X, y)

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
        report = report.replace(f"<= 0.50", "== FALSE")
        report = report.replace(f">  0.50", "== TRUE")
        
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

    records = preprocess_records_for_fitting(records)
    classifier, X, y = fit_to_model(records, target_output)

    generate_output(classifier, X, y)


if __name__ == "__main__":
    main()
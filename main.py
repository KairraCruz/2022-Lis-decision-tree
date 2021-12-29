import pandas as pd
from pprint import pprint
from sklearn import tree, preprocessing

header_consent = "PRIVACY CONSENT. I understand and agree that by filling out this form, I am allowing the researcher (Kairra Cruz) to collect, process, use, share, and disclose my personal information and also to store it as long as necessary for the fulfillment of Facilities Management Capstone Survey of the stated purpose and in accordance with applicable laws, including the Data Privacy Act of 2012 and its Implementing Rules and Regulations. The purpose and extent of the collection, use, sharing, disclosure, and storage of my personal information were cleared to me. "
header_multiple_choices = '6. What kind of service/s do you usually request? You can choose more than 1.'
header_suggestions = "Suggestions to improve existing facilities management processes."

def load_csv():
    filename = "responses.csv"

    return pd.read_csv(filename)

def filter_consent(records):
    # filter out responses that did not answer no

    records_consent_no = records[records[header_consent] != "Yes"]
    print(f"{len(records_consent_no)} in record replied with 'no consent'")

    return records[records[header_consent] == "Yes"]

def filter_suggestions(records):
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
    headers_to_drop = [header_consent, header_suggestions, "Timestamp", "Email Address"]

    return records.drop(headers_to_drop, axis=1)

def preprocess_records_for_fitting(records):
    # preprocess records for fitting to decision tree
    # 1. Transform non-mulitple choice responses with label encoding
    # 2. Transform multiple choices with one-hot encoding

    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder
    # https://stackoverflow.com/a/50259157/1599
    labels = []
    for column in records.columns:
        if column == header_multiple_choices:
            continue

        label = preprocessing.LabelEncoder()
        records[column] = label.fit_transform(records[column].values)
        labels.append((column, label))

    # Transform multiple choices to list
    # Then one-hot encode records
    # https://stackoverflow.com/a/52935270/1599
    # https://stackoverflow.com/a/45312840/1599

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
        
    records[header_multiple_choices] = records[header_multiple_choices].apply(split)
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(sparse_output=True)

    records = records.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(
                records.pop(header_multiple_choices)),
                index=records.index,
                columns=mlb.classes_
            )
    )

    return records, labels

def get_label_from_column(labels, output_column):
    for col, label in labels:
        if col == output_column:
            return label
    return None
    
def main():
    records = load_csv()
    records = filter_consent(records)
    suggestions = filter_suggestions(records)
    records = filter_irrelevant_columns(records)
    records, labels = preprocess_records_for_fitting(records)

    # generate decision tree
    output_column = '26. The overall quality of the work'
    output_label = get_label_from_column(labels, output_column)
    
    X = records.loc[:, records.columns != output_column]
    y = records[output_column]
    classifier = tree.DecisionTreeClassifier().fit(X, y)

    # Graphical tree:
    # https://scikit-learn.org/stable/modules/tree.html#classification
    #tree.plot_tree(classifier)

    # Text tree
    export = tree.export_text(classifier, feature_names=X.columns.tolist())
    
    for i, class_name in enumerate(output_label.classes_):
        export = export.replace(f"class: {i}", f"class: {class_name}")
        
    print(export)

if __name__ == "__main__":
    main()
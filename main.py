import graphviz
import numpy
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn import tree, neighbors, linear_model
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

from dtreeviz.trees import dtreeviz
from pprint import pprint

column_consent = "PRIVACY CONSENT. I understand and agree that by filling out this form, I am allowing the researcher (Kairra Cruz) to collect, process, use, share, and disclose my personal information and also to store it as long as necessary for the fulfillment of Facilities Management Capstone Survey of the stated purpose and in accordance with applicable laws, including the Data Privacy Act of 2012 and its Implementing Rules and Regulations. The purpose and extent of the collection, use, sharing, disclosure, and storage of my personal information were cleared to me."
column_multiple_choices = '6. What kind of service/s do you usually request? You can choose more than 1.'
column_suggestions = "Suggestions to improve existing facilities management processes."
column_have_other_option = '7. How often do you request for the above mentioned services?'
column_have_other_option_values = "Once a month,Twice a month,Once every 3 months,Twice every 3 months,Once every 6 months,Twice every 6 months,Once a year,Twice a year".split(",")
column_have_other_option_subtitute_value = "Other Answer"

columns_irrelevant_dataset = [column_consent, column_suggestions, "Timestamp", "Email Address"]
columns_multiple_choice = [column_multiple_choices]
columns_exclude_one_hot = []
columns_to_label_encode = None      # if None, all columns are encoded

ranking_order = ["Very Dissatisfied", "Dissatisfied", "Satisfied", "Very Satisfied"]
    
target_output = '26. The overall quality of the work'

output_suggestions = "suggestions.txt"

def load_csv():
    filename = "responses.csv"

    return pd.read_csv(filename)

def strip_headers(records):
    return records.rename(columns=lambda x: x.strip())

def filter_consent(records):
    # filter out responses that did not answer no

    records_consent_no = records[records[column_consent] != "Yes"]
    print(f"{len(records_consent_no)} in record replied with 'no consent'")

    return records[records[column_consent] == "Yes"]

def update_other_values(records):
    # one of the column have fill-in-the-blank others option.
    # this caused those rows that have filled in others as extra values instead of an "other" value.
    other_values = set()
    
    def subtitute_func(row):
        if row in column_have_other_option_values:
            return row

        other_values.add(row)
        return column_have_other_option_subtitute_value

    records[column_have_other_option] = records[column_have_other_option].apply(subtitute_func)
    print(f"{len(other_values)} 'other' values found:")
    pprint(other_values)

    return records

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

def get_target_output(records, prompt=True):
    # Ask the user what the target output is
    default_target_output = -1

    for i, column in enumerate(records.columns):
        if column == target_output:
            if prompt:
                print("(default) ", end="")

            default_target_output = i + 1
        
        if prompt:
            print(f"'{column}'")

    choice = ""
    if prompt:
        choice = input("Choose column id to be output (leave empty for default): ").strip()

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
def split_multiple_choice_and_one_hot_encode(records, columns_multiple_choice):
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

def one_hot_encode(records, columns_to_label_encode):
    if columns_to_label_encode is None:
        # if columns_to_label_encode == None, skip encoding columns
        return records

    for column in records.columns:
        if column == target_output or column in columns_to_label_encode:
            continue

        records = one_hot_encode_and_bind(records, column)

    return records

# if columns_to_label_encode == None, encode all columns
def label_encode(records, columns_to_label_encode, target_output, ranking_order):
    # https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
    if columns_to_label_encode is None:
        columns_to_label_encode = records.columns

    class_names = {}

    for column in columns_to_label_encode:
        if column != target_output:
            le = LabelEncoder()
            records[column] = le.fit_transform(records[column])
            class_names[column] = list(le.classes_)
        else:
            le = LabelEncoder()
            le.fit(ranking_order)
            records[column] = le.transform(records[column])
            ranking_order = list(le.classes_)

    return records, ranking_order, class_names

def preprocess_records_for_fitting(records, target_output, ranking_order):
    # preprocess records for fitting to decision tree
    # 1. One hot encode responses with multiple choices
    # 2. Label encode
    # 3. One hot encode responses everything else

    records = split_multiple_choice_and_one_hot_encode(records, columns_multiple_choice)
    records, ranking_order, classes = label_encode(records, columns_to_label_encode, target_output, ranking_order)
    records = one_hot_encode(records, columns_to_label_encode)

    return records, ranking_order, classes

def fit_to_model(classfier_name, classifier_class, records, target_output):
    X = records.loc[:, records.columns != target_output]
    y = records[target_output]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

    classifier = classifier_class.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)
    print(f"{classfier_name} accuracy: {score}")

    return classifier, X, y

def generate_decision_tree_report(classifier, X, y, ranking_order, class_names):
    output_pngfile = "decision_tree.png"
    output_textfile = "decision_tree.txt"
    output_svgfile = "decision_tree.svg"

    def generate_dtreeviz():
        # https://mljar.com/blog/visualize-decision-tree/
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            viz = dtreeviz(classifier, X, y,
                    target_name="target",
                    feature_names=X.columns.tolist(),  
                    #class_names= y.unique(), 
                    class_names= ranking_order, 
            )
        viz.save(output_svgfile) 
        print(f"Output written to {output_svgfile}")

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

    generate_dtreeviz()
    #generate_png()
    #generate_txt()

# https://stackoverflow.com/a/56301555/1599
def generate_density_report(filename, classifier, X, y, ranking_order, class_names):
    '''
    Plot features densities depending on the outcome values
    '''
    # separate data based on outcome values

    data = pd.concat([X, y], axis=1)
    outcome_0 = data[data[target_output] == 0]
    outcome_1 = data[data[target_output] == 1]
    outcome_2 = data[data[target_output] == 2]
    outcome_3 = data[data[target_output] == 3]

    # init figure
    matplotlib.rc("figure", figsize=(20,100))
    
    fig, axs = plt.subplots(len(X.columns), 1)
    fig.suptitle('Features densities')
    plt.subplots_adjust(left = 0.25, right = 0.9, bottom = 0.1, top = 0.95,
                        wspace = 0.2, hspace = 0.9)

    # plot densities for outcomes
    for i, column_name in enumerate(X.columns):
        ax = axs[i]
            
        #plt.subplot(4, 2, names.index(column_name) + 1)
        
        for outcome, color, ranking in zip(
                [outcome_0, outcome_1, outcome_2, outcome_3], 
                ["red", "green", "blue", "yellow"],
                ranking_order
            ):

            try:
                outcome[column_name].plot(kind='density', ax=ax, subplots=True, 
                                            sharex=False, color=color, legend=True,
                                            label=ranking)
                #ax.set_yticklabels(class_names[column_name])
            except numpy.linalg.LinAlgError:
                continue
            
        #ax.set_xlabel(f"Values for '{column_name.strip()}'")
        ax.set_title(f"'{column_name.strip()}' density")
        ax.grid('on')

#    plt.show()
    fig.savefig(filename)

def generate_knn_report(classifier, X, y, ranking_order, class_names):
    filename = "KNN densities.png"
    generate_density_report(filename, classifier, X, y, ranking_order, class_names)
    print(f"Saved {filename}")

def generate_logistic_regression_report(classifier, X, y, ranking_order, class_names):
    filename = "Logistic Regression densities.png"
    generate_density_report(filename, classifier, X, y, ranking_order, class_names)
    print(f"Saved {filename}")

classifiers = [
    ("Decision tree", tree.DecisionTreeClassifier(random_state=42), generate_decision_tree_report),
    ("k-Nearest Neighbors", neighbors.KNeighborsClassifier(), generate_knn_report),
    ("Logistic Regression", linear_model.LogisticRegression(random_state=42, max_iter=1000), generate_logistic_regression_report),
]

def cleanup_dataset(records):
    records = strip_headers(records)
    records = filter_consent(records)
    records = update_other_values(records)
    suggestions = cleanup_suggestions(records)
    records = filter_irrelevant_columns(records)

    return records

def prepare_records(records, ranking_order):
    target_output = get_target_output(records, prompt=False)
    records, ranking_order, class_names = preprocess_records_for_fitting(records, target_output, ranking_order)

    return records, target_output, ranking_order, class_names

def main():
    records = load_csv()
    records = cleanup_dataset(records)

    global ranking_order
    records, target_output, ranking_order, class_names = prepare_records(records, ranking_order)
    
    print("\nScalar values per class:")
    for key, value in class_names.items():
        print(f"\n{key}:")
        for i, v in enumerate(value):
            if v == 0: v = "True"
            if v == 1: v = "False"

            print(f"\t{i}: {v}")

    for classfier_name, classifier_class, classifier_report_generator in classifiers:
        print("")
        classifier, X, y = fit_to_model(classfier_name, classifier_class, records, target_output)

        if classifier_report_generator:
            classifier_report_generator(classifier, X, y, ranking_order, class_names)


def predict(classifier_class_index, input_data_to_predict):
    from collections import OrderedDict

    data_to_predict = OrderedDict()
    for k, v in sorted(input_data_to_predict.items(), key=lambda x: int(x[0][:2].strip("."))):
        data_to_predict[k.strip()] = v.strip()
        print(k)
    
    # temporary add a value in target output to skip making extra code in 
    # prepare_records function for handling NaN values.
    global target_output
    data_to_predict[target_output] = ranking_order[0]

    data_to_predict = pd.DataFrame.from_dict([data_to_predict])

    records = load_csv()
    records = cleanup_dataset(records)
    records_count = len(records)
    records = pd.concat([records, data_to_predict], ignore_index=True)

    records, target_output, ranking_order, class_names = prepare_records(records, ranking_order)

    # after fitting the records, extract the data to predict
    data_to_predict = records[-1:]
    records = records[:-1]
    del data_to_predict[target_output]
    assert len(records) == records_count

    classfier_name, classifier_class, classifier_report_generator = classifiers[classifier_class_index]
    classifier, X, y = fit_to_model(classfier_name, classifier_class, records, target_output)

#    c = records[2:3]
#    print(c)
#    del c[target_output]
#    predict = classifier.predict(c)
#    print(predict)
    predict = classifier.predict(data_to_predict)
#    print(predict)
    output = ranking_order[predict[0]]
    print(output)
    
    return output

if __name__ == "__main__":
    main()

#    data_sample = {'6. What kind of service/s do you usually request? You can choose more than 1.': 'Bidet Installation, Faucet Installation', '1. Location of Condominium/Flat/Apartment/Townhouse/Villa': 'Makati', '2. Gender': 'Male', '3. Age of head of household': '39 - 45', '4. Type of unit': 'Condominium Unit', '5. Number of people in your household': '03-May', '7. How often do you request for the above mentioned services?': 'Once every 3 months', '8. What was the usual status of the request?': 'Not Completed', '9. What was the usual time of the request?': '1:00 pm - 3:00 pm', '10. Over the last month, how many times have you called for maintenance or repairs?': '6 to 10 Times', '11. If you called for NON-EMERGENCY maintenance or repairs (for example, leaky faucet, broken light, etc.) the work was usually completed in:': 'Within 1 day', '12. If you called for EMERGENCY maintenance or repairs (for example, toilet plugged up, gas leak, etc.) the work was usually completed in:': 'Less Than 6 Hours', '17. How did you request the repair service?': 'By Telephone/Mobile', '18. Did you encounter problems when requesting repair service?': 'Sometimes', '19. How did the repair person communicate with you when the repair was completed?': 'He called on the phone', '20. Do you think management provides you information about maintenance and repair (for example, water shut off, modernization activities)?': 'Strongly Agree', '13. How easy it was to request? ': 'Very Satisfied', '14. How well the repairs were done? ': 'Satisfied', '15. Person you contacted? ': 'Satisfied', '16. Your Property Management? ': 'Very Satisfied', '21. Responsive to your questions and concerns? ': 'Satisfied', '22. Being able to arrange a suitable day / date / time for the repair to be carried out': 'Dissatisfied', '23. Time taken before work started': 'Very Satisfied', '24. The speed of completion of the work': 'Very Satisfied', "25. The repair being done 'right first time'": 'Satisfied'}
#    data_sample = {'6. What kind of service/s do you usually request? You can choose more than 1.': 'Bidet Installation, Electric Installation, Heating, Ventilation, and Air Conditioning Basic and General Cleaning, Plumbing Repair', '1. Location of Condominium/Flat/Apartment/Townhouse/Villa': 'Makati', '2. Gender': 'Male', '3. Age of head of household': '18 - 24', '4. Type of unit': 'Flat', '5. Number of people in your household': '03-May', '7. How often do you request for the above mentioned services?': 'Once every 3 months', '8. What was the usual status of the request?': 'Not Completed', '9. What was the usual time of the request?': '10:00 am - 12:00 pm', '10. Over the last month, how many times have you called for maintenance or repairs?': '1 to 5 Times', '11. If you called for NON-EMERGENCY maintenance or repairs (for example, leaky faucet, broken light, etc.) the work was usually completed in:': 'Problem Never Corrected', '12. If you called for EMERGENCY maintenance or repairs (for example, toilet plugged up, gas leak, etc.) the work was usually completed in:': 'More than 24 hours', '17. How did you request the repair service?': 'By Telephone/Mobile', '18. Did you encounter problems when requesting repair service?': 'Very Often', '19. How did the repair person communicate with you when the repair was completed?': 'No one communicated with me', '20. Do you think management provides you information about maintenance and repair (for example, water shut off, modernization activities)?': 'Disagree', '13. How easy it was to request?   ': 'Dissatisfied', '14. How well the repairs were done?   ': 'Dissatisfied', '15. Person you contacted?    ': 'Dissatisfied', '16. Your Property Management?    ': 'Dissatisfied', '21. Responsive to your questions and concerns?    ': 'Dissatisfied', '22. Being able to arrange a suitable day / date / time for the repair to be carried out': 'Dissatisfied', '23. Time taken before work started': 'Dissatisfied', '24. The speed of completion of the work': 'Dissatisfied', "25. The repair being done 'right first time'": 'Satisfied'}
#    predict(0, data_sample)
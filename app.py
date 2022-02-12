from flask import Flask, render_template, url_for, request
from main import predict, ranking_order, classifiers


app = Flask(__name__)

@app.route("/")
def ml_api():
    return render_template("form.html",
                           api_end_point=url_for("ml_api_simulate"))
    
@app.route("/simulate", methods=["POST"])
def ml_api_simulate():
    input_data = map_to_ml_input(request.form)

    outputs = {}

    for classifier_class_index, (classifier_name, *rest) in enumerate(classifiers):
        p = predict(ranking_order, classifier_class_index, input_data)
        outputs[classifier_name] = p

    return str(outputs)


def map_to_ml_input(form):
    multiple_choices_key = "6. What kind of service/s do you usually request? You can choose more than 1."
    number_of_people_in_your_household_key = '5. Number of people in your household'
    
    number_of_people_in_your_household_mapping =  {
        "1 - 2": "01-Feb",
        "3 - 5": "03-May",
        "6 - 7": "06-Jul",
    }

    data = {
        multiple_choices_key: []
    }

    for key, value in form.lists():
        if key != multiple_choices_key:
            value = value[0]

            if key.strip() == number_of_people_in_your_household_key:
                value = number_of_people_in_your_household_mapping.get(value, value)
        else:
            value = ", ".join(value)

        data[key] = value

    if len(data[multiple_choices_key]) == 0:
        data[multiple_choices_key] = ""

    print(data)
    return data
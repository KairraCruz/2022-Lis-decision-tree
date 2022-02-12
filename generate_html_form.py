import pandas as pd

choices = [
('1. Location of Condominium/Flat/Apartment/Townhouse/Villa', "Makati,Manila,Taguig,Quezon City,Outside Manila,Outside the Philippines".split(",")),
('2. Gender', "Male,Female".split(",")),
('3. Age of head of household', "18 - 24,25 -31,32 - 38,39 - 45,46 - 52,53 - 59,60 and above".split(",")),
('4. Type of unit', "Apartment,Condominium Unit,Flat,Townhouse,Villa".split(",")),
('5. Number of people in your household', "1 - 2,3 - 5,6 - 7,8 and above".split(",")),
('7. How often do you request for the above mentioned services?', "Once a month,Twice a month,Once every 3 months,Twice every 3 months,Once every 6 months,Twice every 6 months,Once a year,Twice a year,Other Answer".split(",")),
('8. What was the usual status of the request?', "Completed,Not Completed".split(",")),
('9. What was the usual time of the request?', "8:00 am - 10:00 am,10:00 am - 12:00 pm,1:00 pm - 3:00 pm,3:00 pm - 5:00pm".split(",")),
('10. Over the last month, how many times have you called for maintenance or repairs?', "Have Never Called,1 to 5 Times,6 to 10 Times,More Than 10 Times".split(",")),
('11. If you called for NON-EMERGENCY maintenance or repairs (for example, leaky faucet, broken light, etc.) the work was usually completed in:', "Within 1 day,Within 2 days,Within 3 days,Within 4 days,5 days or more,Problem Never Corrected".split(",")),
('12. If you called for EMERGENCY maintenance or repairs (for example, toilet plugged up, gas leak, etc.) the work was usually completed in:', "Have Never Called,Less Than 6 Hours,6 to 24 hours,More than 24 hours,Problem Never Corrected".split(",")
),
('17. How did you request the repair service?', "By Telephone/Mobile,By Email,With the help of the Administration Office (Walk-In)".split(",")),
('18. Did you encounter problems when requesting repair service?', "Always,Very Often,Sometimes,Rarely,Never".split(",")),
('19. How did the repair person communicate with you when the repair was completed?', "We spoke in person,He called on the phone,A note was left in my unit,No one communicated with me".split(",")),
('20. Do you think management provides you information about maintenance and repair (for example, water shut off, modernization activities)?', "Strongly Agree,Agree,Does not apply,Disagree,Strongly Disagree".split(",")),
]

statisfactions = [
('13. How easy it was to request?   ' ),
('14. How well the repairs were done?   ' ),
('15. Person you contacted?    ' ),
('16. Your Property Management?    ' ),
('21. Responsive to your questions and concerns?    ' ),
('22. Being able to arrange a suitable day / date / time for the repair to be carried out' ),
('23. Time taken before work started' ),
('24. The speed of completion of the work' ),
('25. The repair being done \'right first time\'' ),
]

checkboxes =[
'Bidet Installation',
'Ceiling Fan Installation and Repair',
'Cook Top, Range and Stove Installation and Repair',
'Custom Made Cabinets and Installation', 'Door Repair', 'Double Locks',
'Drain Cleaning', 'Drywall Repair', 'Electric Installation',
'Electric Repair', 'Exhaust Fan Installation and Repair',
'Faucet Installation', 'Heating, Ventilation, and Air Conditioning',
'Heating, Ventilation, and Air Conditioning Basic and General Cleaning',
'Heating, Ventilation, and Air Conditioning Installation',
'Lighting and Fixtures Installation and Repair',
'Other problems encountered with the facilities management process that were not mentioned in the items above',
'Other services not mentioned above', 'Plumbing Installation',
'Plumbing Repair', 'Shower Heater installation',
'TV/Bracket Installation and Repair',
'Washing Machine Installation and Repair', 'Water Closet Bowl']

url = "{{ api_end_point }}"
output = f"""
<style>
label
{{
    display: block;
}}

div#spacer
{{
    margin-top: 30px;
}}

</style>

<form action="{url}" method="post">
"""

template = """
    <label>{}: 
        <input id="{}" type="radio" name="{}" value="{}" required>
    </label>
"""

for name, choice_values in choices:
    id = name.split(".")[0]

    output += f"<label>{name}</label>"

    for value in choice_values:
        output += template.format(value, id, name, value)

    output += "<div id=spacer></div>"

template = """
    <label>{}: 
        <input id="{}" type="radio" name="{}" value="{}" required>
    </label>
"""

for name in statisfactions:
    id = name.split(".")[0]
    output += f"<label>{name}</label>"

    for label in ["Very Satisfied", "Satisfied", "Dissatisfied", "Very Dissatisfied"]:
        output += template.format(label, id, name, label)

    output += "<div id=spacer></div>"

id = "6. What kind of service/s do you usually request? You can choose more than 1."
output += f"<label>{id}</label>"
template = """
    <label>{}:
        <input id="{}" type="checkbox" name="{}" value="{}" >
    </label>
"""

for name in checkboxes:
    output += template.format(name, id, id, name)

output += """
    <input type="submit" value="predict">
</form>"""

print(output)
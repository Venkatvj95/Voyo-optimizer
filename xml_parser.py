import xml.etree.ElementTree as ET

tree = ET.parse("data/process.xml")
root = tree.getroot()

activities = []

for activity in root.findall("Activity"):

    activities.append({
        "name": activity.find("Name").text,
        "duration": activity.find("Duration").text,
        "cost": activity.find("Cost").text,
        "resource": activity.find("Resource").text
    })

print(activities)
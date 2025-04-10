import pandas as pd
from pathlib import Path


def load_nts(
    trips_path: Path,
    individuals_path: Path,
    households_path: Path,
    years=None,
):
    trips = load_trips(trips_path, years=years)
    people = load_individuals(individuals_path, years=years)
    households = load_hhs(households_path, years=years)

    # join individuals with households
    people_hids = set(people.hid)
    hhs_hids = set(households.index)

    if people_hids != hhs_hids:
        print("HIDs in people and households do not match, attempting to fix...")
        n_people_hids = len(people_hids)
        n_hhs_hids = len(hhs_hids)
        combined_hids = list(people_hids & hhs_hids)
        people = people[people.hid.isin(combined_hids)]
        households = households.loc[combined_hids]
        print(
            f"Fixed: People {n_people_hids} -> {len(people.hid)}, HHs {n_hhs_hids} -> {len(households.index)}"
        )

    labels = people.join(households.drop(columns=["year"]), on="hid")

    # check unique iids
    assert labels.index.is_unique

    trips_iids = set(trips.iid)
    labels_iids = set(labels.index)

    if trips_iids != labels_iids:
        print("IIDs in trips and labels do not match, attempting to fix...")
        n_trip_iids = len(trips_iids)
        n_label_iids = len(labels_iids)
        combined_iids = list(trips_iids & labels_iids)
        trips = trips[trips.iid.isin(combined_iids)]
        labels = labels.loc[combined_iids]
        print(
            f"Fixed: Trips {n_trip_iids} -> {len(trips.iid)}, Labels {n_label_iids} -> {len(labels.index)}"
        )

    assert trips_iids == labels_iids

    return trips, labels


def load_trips(path, years=None):
    data = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "TripID",
            "JourSeq",
            "DayID",
            "IndividualID",
            "HouseholdID",
            "MainMode_B04ID",
            "TripPurpFrom_B01ID",
            "TripPurpTo_B01ID",
            "TripStart",
            "TripEnd",
            "TripOrigGOR_B02ID",
            "TripDestGOR_B02ID",
            "W5",
            "SurveyYear",
        ],
    )
    data = data.rename(
        columns={
            "TripID": "tid",
            "JourSeq": "seq",
            "DayID": "day",
            "IndividualID": "iid",
            "HouseholdID": "hid",
            "TripOrigGOR_B02ID": "ozone",
            "TripDestGOR_B02ID": "dzone",
            "TripPurpFrom_B01ID": "oact",
            "TripPurpTo_B01ID": "dact",
            "MainMode_B04ID": "mode",
            "TripStart": "tst",
            "TripEnd": "tet",
            "W5": "freq",
            "SurveyYear": "year",
        }
    )

    if years:
        data = data[data.year.isin(years)]

    data.tst = pd.to_numeric(data.tst, errors="coerce")

    data.tet = pd.to_numeric(data.tet, errors="coerce")
    data.ozone = pd.to_numeric(data.ozone, errors="coerce")
    data.dzone = pd.to_numeric(data.dzone, errors="coerce")
    data.freq = pd.to_numeric(data.freq, errors="coerce")

    data["did"] = data.groupby("iid")["day"].transform(lambda x: pd.factorize(x)[0])
    data["pid"] = data["hid"].astype(str) + "_" + data["iid"].astype(str)
    data = data.loc[
        data.groupby("pid").filter(lambda x: pd.isnull(x).sum().sum() < 1).index
    ]
    # remove wrapping of end times around midnight
    data.loc[data.tet == 0, "tet"] = 1440
    data.loc[data.tet < data.tst, "tet"] += 1440

    mode_mapping = {
        1: "walk",
        2: "bike",
        3: "car",  #'Car/van driver'
        4: "car",  #'Car/van driver'
        5: "car",  #'Motorcycle',
        6: "car",  #'Other private transport',
        7: "bus",  # Bus in London',
        8: "bus",  #'Other local bus',
        9: "bus",  #'Non-local bus',
        10: "train",  #'London Underground',
        11: "train",  #'Surface Rail',
        12: "taxi",  #'Taxi/minicab',
        13: "pt",  #'Other public transport',
        -10: "DEAD",
        -8: "NA",
    }

    purp_mapping = {
        1: "work",
        2: "work",  #'In course of work',
        3: "education",
        4: "shop",  #'Food shopping',
        5: "shop",  #'Non food shopping',
        6: "medical",  #'Personal business medical',
        7: "other",  #'Personal business eat/drink',
        8: "other",  #'Personal business other',
        9: "social",  #'Eat/drink with friends',
        10: "social",  #'Visit friends',
        11: "social",  #'Other social',
        12: "social",  #'Entertain/ public activity',
        13: "social",  #'Sport: participate',
        14: "hotel",  #'Holiday: base',
        15: "other",  #'Day trip/just walk',
        16: "other",  #'Other non-escort',
        17: "escort",  #'Escort home',
        18: "escort",  #'Escort work',
        19: "escort",  #'Escort in course of work',
        20: "escort",  #'Escort education',
        21: "escort",  #'Escort shopping/personal business',
        22: "escort",  #'Other escort',
        23: "home",  #'Home',
        -10: "DEAD",
        -8: "NA",
    }

    data["mode"] = data["mode"].map(mode_mapping)
    data["oact"] = data["oact"].map(purp_mapping)
    data["dact"] = data["dact"].map(purp_mapping)
    data.tst = data.tst.astype(int)
    data.tet = data.tet.astype(int)
    return data


def load_individuals(path: Path, years=None):
    columns = {
        "SurveyYear": "year",
        "IndividualID": "iid",
        "HouseholdID": "hid",
        "Age_B01ID": "age",
        "Sex_B01ID": "gender",
        "EdAttn1_B01ID": "education",
        "DrivLic_B02ID": "license",
        "CarAccess_B01ID": "car_access",
        "EcoStat_B02ID": "work_status",
        "EthGroupTS_B02ID": "ethnicity",
    }
    attributes = pd.read_csv(path, sep="\t", usecols=columns.keys()).rename(
        columns=columns
    )

    if years:
        attributes = attributes[attributes.year.isin(years)]

    # fix special values to zero (DEAD, NULL, NA, etc)
    for c in [
        "age",
        "gender",
        "education",
        "license",
        "car_access",
        "work_status",
        "ethnicity",
    ]:
        attributes.loc[attributes[c] < 0, c] = 0
        attributes.loc[attributes[c].isnull(), c] = 0

    return attributes.set_index("iid")


def load_hhs(path: Path, years=None):
    columns = {
        "HouseholdID": "hid",
        "Settlement2011EW_B04ID": "area",
        "SurveyYear": "year",
        "HHIncQISEngTS_B01ID": "income",
        "HHoldNumPeople": "hh_size",
        "HHoldStruct_B02ID": "hh_composition",
        "HHoldNumChildren": "hh_children",
        "NumCar": "hh_cars",
        "NumBike": "hh_bikes",
        "NumMCycle": "hh_motorcycles",
    }
    hhs = pd.read_csv(path, sep="\t", usecols=columns.keys()).rename(columns=columns)

    if years:
        hhs = hhs[hhs.year.isin(years)]

    for c in [
        "area",
        "income",
        "hh_size",
        "hh_composition",
        "hh_children",
        "hh_cars",
        "hh_bikes",
        "hh_motorcycles",
    ]:
        hhs.loc[hhs[c] < 0, c] = 0
        hhs.loc[hhs[c].isnull(), c] = 0

    return hhs.set_index("hid")


def label_mapping(labels):

    labels_copy = labels.copy()

    age_mapping = {
        1: "<5",
        2: "<5",
        3: "<5",
        4: "5-11",
        5: "11-16",
        6: "16-20",
        7: "16-20",
        8: "16-20",
        9: "16-20",
        10: "20-30",
        11: "20-30",
        12: "20-30",
        13: "30-40",
        14: "40-50",
        15: "50-70",
        16: "50-70",
        17: "50-70",
        18: "70+",
        19: "70+",
        20: "70+",
        21: "70+",
    }
    gender_mapping = {0: "unknown", 1: "M", 2: "F"}
    education_mapping = {0: "unknown", 1: "Y", 2: "N"}
    license_mapping = {0: "unknown", 1: "yes", 2: "yes", 3: "no"}
    car_access_mapping = {
        0: "unknown",
        1: "yes",
        2: "yes",
        3: "yes",
        4: "yes",
        5: "no",
        6: "no",
    }
    work_status_mapping = {
        0: "unemployed",
        1: "employed",
        2: "employed",
        3: "unemployed",
        4: "unemployed",
        5: "student",
        6: "unemployed",
    }
    area_mapping = {
        0: "unknown",
        1: "suburban",
        2: "urban",
        3: "rural",
        4: "rural",
        5: "scotland",
    }
    ethnicity_mapping = {0: "unknown", 1: "white", 2: "non-white"}
    hh_composition_mapping = {
        0: "unknown",
        1: "1adult",
        2: "2adults",
        3: "3+adults",
        4: "single_parent",
        5: "2adult_1+child",
        6: "3+adult_1+child",
    }

    mappings = {
        "age": age_mapping,
        "gender": gender_mapping,
        "education": education_mapping,
        "license": license_mapping,
        "car_access": car_access_mapping,
        "work_status": work_status_mapping,
        "area": area_mapping,
        "ethnicity": ethnicity_mapping,
        "hh_composition": hh_composition_mapping,
    }

    for c, mapping in mappings.items():
        if c in labels_copy.columns:
            labels_copy[c] = labels_copy[c].map(mapping)

    return labels_copy

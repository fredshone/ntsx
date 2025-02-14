import pandas as pd


def trips(path, years=None):
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

    data["did"] = data.groupby("iid")["day"].transform(lambda x: pd.factorize(x)[0] + 1)
    data["pid"] = data["hid"].astype(str) + "_" + data["iid"].astype(str)
    data = data.loc[
        data.groupby("pid").filter(lambda x: pd.isnull(x).sum().sum() < 1).index
    ]
    data.loc[data.tet == 0, "tet"] = 1440

    # travel_diaries = travel_diaries.drop(["tid", "day", "year", "did"], axis=1)

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

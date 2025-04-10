from ntsx.encoders.trip_encoder import TripEncoder
from pandas import DataFrame


def test_graph_encoder():
    trips = DataFrame(
        {
            "mode": ["car", "car", "bike", "bike"],
            "oact": ["home", "work", "home", "shop"],
            "dact": ["work", "home", "shop", "home"],
            "day": [0, 0, 0, 0],
            "tst": [360, 720, 1000, 1200],
            "tet": [700, 800, 1030, 1230],
            "ozone": ["zone1", "zone2", "zone1", "zone3"],
            "dzone": ["zone2", "zone1", "zone3", "zone1"],
        }
    )
    encoder = TripEncoder(trips)
    assert encoder.get_encoding("mode") == "categorical"
    assert encoder.get_encoding("oact") == "categorical"
    assert encoder.get_encoding("dact") == "categorical"
    assert encoder.get_encoding("day") == "categorical"
    assert encoder.get_encoding("tst") == "time"
    assert encoder.get_encoding("tet") == "time"
    assert encoder.get_encoding("ozone") == "categorical"
    assert encoder.get_encoding("dzone") == "categorical"
    assert encoder.get_size("mode") == 2
    assert encoder.get_size("oact") == 3
    assert encoder.get_size("dact") == 3
    assert encoder.get_size("day") == 1
    assert encoder.get_size("tst") == 1
    assert encoder.get_size("tet") == 1
    assert encoder.get_size("ozone") == 3
    assert encoder.get_size("dzone") == 3

    assert encoder.encode("car", "mode") == 1
    assert encoder.decode(1, "mode") == "car"

    assert encoder.encode(1440, "tet") == 1.0
    assert encoder.decode(1.0, "tet") == 1440

from arpes.utilities.qt import QtInfo


def test(actual, expected):
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_QtInfo():
    info = QtInfo()
    test(info.inches_to_px(1), 150)
    test(info.inches_to_px(0.5), 75)
    test(list(info.inches_to_px([1, 0.5])), [150, 75])


def run_all_tests():
    test_QtInfo()


if __name__ == "__main__":
    run_all_tests()

def to_absolute_month(month, year):
    return month + 12 * (year - 2015) - 1


assert to_absolute_month(1, 2015) == 0
assert to_absolute_month(2, 2015) == 1
assert to_absolute_month(12, 2015) == 11
assert to_absolute_month(1, 2016) == 12
assert to_absolute_month(10, 2016) == 21
assert to_absolute_month(4, 2017) == 27

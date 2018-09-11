import csv


class Data(object):

    def __init__(self, number_of_cities):
        with open("european_cities.csv", "r") as f:
            data = list(csv.reader(f, delimiter=';'))
            city_list = data[0]
            distance_matrix = [[float(d) for d in row]
                               for row in data[1:]]
        n = number_of_cities

        self.cities = city_list[:n]
        self.distances = [row[:n] for row in distance_matrix[:n]]

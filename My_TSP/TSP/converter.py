import math


prova = []

with open('/home/fabrizio/Scrivania/Much-Cross-Little-Over/My_TSP/Istances/dantzig42.txt') as file:
    for line in file.readlines():
        l = line.rstrip()
        p1 = float(l.split(",")[0])
        p2 = float(l.split(",")[1])
        prova.append((p1, p2))

r = 6371  # KM
phi_0 = prova[0][1]
cos_phi_0 = math.cos(math.radians(phi_0))


def to_xy(point, r, cos_phi_0):
    lam = point[0]
    phi = point[1]
    return (r * math.radians(lam) * cos_phi_0, r * math.radians(phi))


with open('/home/fabrizio/Scrivania/Much-Cross-Little-Over/My_TSP/Istances-Converter/dantzig42.txt', 'w') as file:
    for point in prova:
        point_xy = to_xy(point, r, cos_phi_0)
        file.write(str(point_xy[0])+','+str(point_xy[1])+'\n')

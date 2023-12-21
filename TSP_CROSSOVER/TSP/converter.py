import math

'''
    This script is use to convert the lat,lng coordinate in x,y coordinate. 
    
    For Transformation: 
        - Create list with all lat,lng point 
        - Calcolate function to transoform lat, lng in xy where: 
            x = r λ cos(φ0)
            y = r φ
        And after return the value. 

'''

# Inizialize list of all_point
all_point = []

# Read file txt and point by istances with coordinates lat,lng
with open('/home/fabrizio/Scrivania/Much-Cross-Little-Over/My_TSP/Istances/ulysses22.txt') as file:
    for line in file.readlines():
        l = line.rstrip()
        p1 = float(l.split(",")[0])
        p2 = float(l.split(",")[1])
        all_point.append((p1, p2))

# Radio earth
r = 6378  # KM

# Calcolate φ
phi_0 = all_point[0][1]

# Calcolate cos(φ0)
cos_phi_0 = math.cos(math.radians(phi_0))


# Function to calcolate xy coordinates
def to_xy(point, r, cos_phi_0):
    lam = point[0]
    phi = point[1]
    return (r * math.radians(lam) * cos_phi_0, r * math.radians(phi))


# Write convert result in a txt file
with open('/home/fabrizio/Scrivania/Much-Cross-Little-Over/My_TSP/Istances-Converter/ulysses22.txt', 'w') as file:
    for point in all_point:
        point_xy = to_xy(point, r, cos_phi_0)
        file.write(str(point_xy[0])+','+str(point_xy[1])+'\n')

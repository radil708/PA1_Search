# NORTH = 'North'
# SOUTH = 'South'
# EAST = 'East'
# WEST = 'West'
# STOP = 'Stop'
#
# LEFT = {NORTH: WEST,
#                    SOUTH: EAST,
#                    EAST:  NORTH,
#                    WEST:  SOUTH,
#                    STOP:  STOP}
#
# for x,y in LEFT.items():
#     print(y,x)
#
# RIGHT = dict([(y,x) for x, y in LEFT.items()])
#
# print(RIGHT)

initialValue=False
height = 10
width = 5
x = [[initialValue for y in range(height)] for x in range(width)]
#print(x)
for each in x:
    print(each)
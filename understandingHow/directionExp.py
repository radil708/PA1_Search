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
import heapq

heap = []
count = 0
priority_0 = 0
priority_1 = 1
priority_2 = 2

entry = (priority_2, count, "A")
heapq.heappush(heap, entry)
count+=1

entry = (priority_1, count, "B")
heapq.heappush(heap, entry)
count+=1

entry = (priority_0, count, "V")
heapq.heappush(heap, entry)
count+=1

print(heap)
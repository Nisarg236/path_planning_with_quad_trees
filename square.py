from itertools import count

#Define the Square class
class Square:
    _ids = count(1)  # Counter for generating unique IDs
    def __init__(self, top_left, bottom_right):
        self.id = next(self._ids)
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.length = bottom_right[1] - top_left[1]
        self.width = bottom_right[0] - top_left[0]
        self.center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def to_dict(self):
        return {
            'id': self.id,
            'top_left': self.top_left,
            'bottom_right': self.bottom_right,
            'length': self.length,
            'width': self.width,
            'center': self.center,
            'neighbors': [n.id for n in self.neighbors]
        }
    

from math import sqrt

def drawCircle(pos, imageArray, r=2, color=255):
    xmin = max(pos[0] - r, 0)
    xmax = min(pos[0] + r + 1, imageArray.shape[0])
    for x in range(xmin, xmax):
        dx = abs(pos[0] - x)
        dy = int(sqrt(r ** 2 - dx ** 2))
        ymin = max(pos[1] - dy, 0)
        ymax = min(pos[1] + dy + 1, imageArray.shape[1])
        for y in range(int(ymin), int(ymax)):
            imageArray[x, int(y)] = color

def drawLine(pos, rel, imageArray, color, width=1):
    """
    Using a naive line-drawing algorithm for now
    """
    # TODO: Check https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    x1 = (pos[0] - rel[0]) // 2
    x2 = (pos[0]) // 2
    y1 = (pos[1] - rel[1]) // 2
    # y2 = (pos[1]) // 2
    dx = (rel[0]) // 2
    dy = (rel[1]) // 2
    x_start = x1
    if x1 > x2:
        x1, x2 = x2, x1
    for x in range(x1, x2 + 1):
        if dx == 0:
            y = y1
        else:
            y = y1 + dy * (x - x_start) / dx
        drawCircle([x, y], imageArray, r=width, color=color)
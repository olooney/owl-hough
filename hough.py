import sys
import math
from PIL import Image, ImageDraw
from glob import glob
from math import sin, cos, pi, sqrt
import os.path

SEARCH_GRID = 8
CLIMB_RANGE = 10
RHO_STRETCH = 2
THETA_SIZE = 180
MAX_PEAKS = 5
CUTOFF = 5

# TODO: theta coordinate really should wrap instead of being cut off
def out_of_bounds(img, point):
    if point[0] < 0 or point[1] < 0:
        return True
    if point[0] >= img.size[0]:
        return True
    if point[1] >= img.size[1]:
        return True
    return False


def gausian_bump(img, center):
    x, y = center

    def bump_pixel(point, vote):
        if not out_of_bounds(img, point):
            old_value = img.getpixel(point)
            img.putpixel(point, old_value + vote)

    bump_pixel( (x,y), 9)

    bump_pixel( (x+1,y), 5)
    bump_pixel( (x-1,y), 5)
    bump_pixel( (x,y+1), 5)
    bump_pixel( (x,y-1), 5)

    bump_pixel( (x+1,y+1), 2)
    bump_pixel( (x-1,y-1), 2)
    bump_pixel( (x+1,y-1), 2)
    bump_pixel( (x-1,y+1), 2)


def hough_transform(img):

    width, height = img.size

    # equation: rho = x cos(theta) + y sin(theta)


    # acc is the accumulator, the hough transformed space
    mode = 'I'
    offset = int(max(height, width) * RHO_STRETCH * sqrt(2) +1)
    rho_size = offset * 2 + 1
    transform = Image.new(mode, (rho_size, THETA_SIZE), 0)

    for x in xrange(width):
        for y in xrange(height):
            if img.getpixel((x,y)) < 192:
                for theta in xrange(THETA_SIZE):
                    radians = pi * theta / THETA_SIZE 
                    rho = x * cos(radians) + y * sin(radians) 
                    point = (int(round(rho*RHO_STRETCH) + offset+1), theta)
                    gausian_bump(transform, point)
                    # vote = transform.getpixel(point) + 10
                    #print repr( (x,y) ), repr(point), vote
                    #transform.putpixel(point, vote) # TODO: weighting, overflow

    # todo: peek detection
    return transform

def climb_hill(img, start):
    ''' starting from a point, climbs to the nearest local maxima '''
    point = start
    previous_point = None
    value = img.getpixel(start)

    # scan right until we find a non-zero value
    while value == 0:
        point = ( (point[0]+1) % img.size[0], point[1])
        if point == start:
            return value, point
        value = img.getpixel(point)
            
    # climb the hill by moving to the highest neighbor pixel
    # until a local maxima is reached.
    climb_range = range(-CLIMB_RANGE, CLIMB_RANGE + 1)
    while point != previous_point:
        previous_point = point
        for dx in climb_range:
            for dy in climb_range:
                if dx == 0 and dy == 0:
                    continue
                essay = (point[0]+dx, point[1]+dy)

                if essay[1] >= img.size[1]:
                    # mobious topology
                    essay = (img.size[0] - essay[0], essay[1] % img.size[1])

                if not out_of_bounds(img, essay):
                    essay_value = img.getpixel(essay)
                    if essay_value > value:
                        point = essay
                        value = essay_value

    # consider nearby contribution too
    # for dx in climb_range:
    #     for dy in climb_range:
    #         if not (dx == 0 and dy == 0):
    #             neighbor = (point[0]+dx, point[1]+dy)
    #             if not out_of_bounds(img, neighbor):
    #                 raw_value = img.getpixel(neighbor)
    #                 value += float(raw_value) / (dx**2 + dy**2)

    return int(value), point
    

def find_peaks(img, subdivisions):
    peaks = {}

    dx = img.size[0] // subdivisions
    dy = img.size[1] // subdivisions

    for i in xrange(subdivisions):
        for j in xrange(subdivisions):
            start = (dx * i, dy * j)
            peak, value  = climb_hill(img, start)
            peaks[peak] = value

    return peaks.items()


def most_prominent(peaks):
    ''' limits peaks to only those in the top half. '''

    if len(peaks) <= 1:
        return peaks
    values = [ value for value, peak in peaks ]
    threshold = max(values) / CUTOFF
    peaks = sorted([ (v, p) for v, p in peaks if v > threshold ], reverse=True)
    
    # TODO: collapse nearby peaks to a single value. I can't
    # implement this until I do segmenting, though.
    
    return peaks[:MAX_PEAKS]
    


def detect(filename):
    """ applies the hough transform to detect lines,
        reduces those to segments, then saves a new
        image with the segments draw on top of the original."""

    input_filename = os.path.join('problems', filename)
    output_filename = os.path.join('solutions', filename)

    img = Image.open(input_filename).convert("L")
    transform = hough_transform(img)

    peaks = most_prominent(find_peaks(transform, SEARCH_GRID))

    output = transform.convert('RGB')
    draw_output = ImageDraw.Draw(output)
    for index, (value, peak)  in enumerate(peaks):
        print filename, repr(peak), value
        for radius in range(3,4):
            if index < 3:
                color = (255,0,0)
            else:
                color = (0,0,200)
            draw_output.ellipse(
                (peak[0]-radius, peak[1]-radius, 
                 peak[0]+radius, peak[1]+radius), 
                outline=color)
                #outline=(128+value//5, value//5, value//5))
    output.save(output_filename)
    #save_transform(input_filename, transform, output_filename)


if __name__ == '__main__':
    # solve the four test mazes
    problem_filter = sys.argv[1] if len(sys.argv) > 1 else ''

    def solve_if(name):
        if name.startswith(problem_filter):
            detect(name)
    
    for path in glob('problems/*.*'):
        name = os.path.basename(path)
        solve_if(name)

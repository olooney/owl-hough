'''
Experiment to apply a straight-line Hough transform to an image and determine
the resulting features. It loops over the images in the problems folder and
generates a corresponding image of the hough transform annotated with detected
peaks in the solutions folder.

http://en.wikipedia.org/wiki/Hough_transform
'''

import sys
import math
from collections import Counter
from PIL import Image, ImageDraw, ImageFilter
from glob import glob
from math import sin, cos, tan, pi, sqrt, floor
import os.path

SEARCH_GRID = 8
CLIMB_RANGE = 10
RHO_STRETCH = 2
THETA_SIZE = 180

PEAK_RADIUS = 5
PEAK_THRESHOLD = 90 # needs 10 pixels contributing

SEGMENT_FILL_GAP = 3
SEGMENT_MIN_LENGTH = 7


def mobius_wrap(size, point):
    # mobius topology
    if point[1] >= size[1]:
        return (size[0] + 1 - point[0], point[1] % size[1])
    elif point[1] < 0:
        return (size[0] + 1 - point[0], size[1] + point[1])
    else:
        return point

def rho_offset(img):
    width, height = img.size
    return int(max(height, width) * RHO_STRETCH * sqrt(2) + 1)

# TODO: theta coordinate really should wrap instead of being cut off
def out_of_bounds(img, point):
    if point[0] < 0 or point[1] < 0:
        return True
    if point[0] >= img.size[0]:
        return True
    if point[1] >= img.size[1]:
        return True
    return False


def brighten(img, center):
    x, y = center

    def bump_pixel(point, vote):
        point = mobius_wrap(img.size, point)
        if not out_of_bounds(img, point):
            old_value = img.getpixel(point)
            img.putpixel(point, old_value + vote)

    bump_pixel((x, y), 9)

    bump_pixel((x+1, y+0), 5)
    bump_pixel((x-1, y+0), 5)
    bump_pixel((x+0, y+1), 5)
    bump_pixel((x+0, y-1), 5)

    bump_pixel((x+1, y+1), 2)
    bump_pixel((x-1, y-1), 2)
    bump_pixel((x+1, y-1), 2)
    bump_pixel((x-1, y+1), 2)


def hough_transform(img):
    '''transform the image to the straight-line hough space of the given image
    as 32-bit integer greyscale image.'''

    # equation: rho = x cos(theta) + y sin(theta)

    width, height = img.size

    # rho naturally goes negative and is centered on zero (for lines going
    # through the origin) but the transform image can only handle positive
    # coordinates, so we shift it so that rho = 0 actually lights up a pixel in
    # the centerline of the transform image.
    offset = rho_offset(img)
    rho_size = offset * 2 + 1

    # Modes: http://svn.effbot.org/public/tags/pil-1.1.4/libImaging/Unpack.c
    mode = 'I'

    # start with an empty transform space; the algorithm will accumulate
    # information into it as it goes. The x-axis is rho, y-axis is theta.
    transform = Image.new(mode, (rho_size, THETA_SIZE), 0)

    for x in xrange(width):
        for y in xrange(height):
            if img.getpixel((x, y)) < 192:
                for theta in xrange(THETA_SIZE):
                    radians = pi * theta / THETA_SIZE

                    # this is where the magic happens
                    rho = x * cos(radians) + y * sin(radians)

                    point = (int(round(rho*RHO_STRETCH) + offset+1), theta)
                    brighten(transform, point)

    return transform


def find_peaks(img):

    peaks = []

    radius_squared = PEAK_RADIUS ** 2
    xy_range = range(-int(floor(PEAK_RADIUS)), int(floor(PEAK_RADIUS))+1)

    for cx in xrange(img.size[0]):
        for cy in xrange(img.size[1]):
            center = (cx, cy)
            center_height = img.getpixel(center)
            highest_neighbor = 0
            total_neighbor_height = 0
            total_neighbor_pixels = 0
            for dx in xy_range:
                for dy in xy_range:
                    neighbor = mobius_wrap(img.size, (cx+dx, cy+dy))

                    if not (dx == 0 and dy == 0 ):
                        if not out_of_bounds(img, neighbor):
                            if (dx**2 + dy**2) < radius_squared:
                                height = img.getpixel(neighbor)

                                # increment these for the average
                                total_neighbor_height += height
                                total_neighbor_pixels += 1

                                # also pick off the maximum
                                if height > highest_neighbor:
                                    highest_neighbor = height

            # only include as a peak if it's a clear local maxima
            if center_height > highest_neighbor:
                average_neighbor_height = floor(total_neighbor_height / total_neighbor_pixels)
                relative_height = center_height - average_neighbor_height
                if relative_height > PEAK_THRESHOLD:
                    # print 'peaks2: %d at (%d, %d)' % (relative_height, cx, cy)
                    #peaks.putpixel(center, relative_height)
                    peaks.append((relative_height, center))

    return peaks


def describe(peak):
    '''attempts to describe a peak in human understandable terms'''
    strength, (rho, theta) = peak
    rough_angle = int(round(float(theta)/45.0))
    direction = [
        'vertical',
        'upwards diagonal',
        'horizontal',
        'downwards diagonal',
        'vertical'
    ][rough_angle]

    return direction + ' stroke'

    # TODO: this doesn't work at all, needs to be smarter about directions
    if direction == 'horizontal':
        positions = ['top', 'middle', 'bottom']
    else:
        positions = ['left', 'center', 'right']

    # TODO: make rho threshold dynamic
    rho_size = 170.0
    position_index = int(floor(3.0 * rho / rho_size))
    position = positions[position_index]

    return ' '.join([position, direction, 'stroke'])


def find_segments(img, peak):
    '''returns endpoints for the segments in the line that are actually active).
    Several hueristic are used to detect segments. 1st, because the matched line
    may not perfectly align with the original image, we check the 3x3 grid of pixels
    surrounded on the point. 2nd, we allow gaps up to a certain threshold; if another
    pixel is encountered before the gap runs out, the segment continues. This allows
    for dotted/dashed lines or simply missing an occasional pixel.  Third, only
    segments above a certain length will be returned. This avoids returning segments
    for spurious points that happen to be in alignment with the detected feature.
    '''

    height, (shifted_rho, theta) = peak

    radians = pi * theta / THETA_SIZE
    # point = (int(round(rho*RHO_STRETCH) + offset+1), theta)
    rho = int(round((shifted_rho - rho_offset(img) - 1)/RHO_STRETCH))

    if theta < 45 or theta > 135:
        # line is more vertical, iterate from top to bottom
        equation = 'x = m y + b'
        slope = -sin(radians)/cos(radians)
        intercept = rho / cos(radians)

        def points_in_line():
            y_range = xrange(0, img.size[1])
            for y in y_range:
                x = slope * y + intercept
                point = (int(x), int(y))
                if not out_of_bounds(img, point):
                    yield point
    else:
        # line is more horizontal, iterate from left to right
        equation = 'y = m x + b'
        slope = -cos(radians)/sin(radians)
        intercept = rho / sin(radians)

        def points_in_line():
            x_range = xrange(0, img.size[0])
            for x in x_range:
                y = slope * x + intercept
                point = (int(x), int(y))
                if not out_of_bounds(img, point):
                    yield point

    def sample_pixel(point):
        point = mobius_wrap(img.size, point)
        if not out_of_bounds(img, point):
            return img.getpixel(point)
        else:
            return 255

    def sample_region(point):
        x, y = point
        return min((
            sample_pixel((x+0, y+0)),
            sample_pixel((x+1, y+0)),
            sample_pixel((x-1, y+0)),
            sample_pixel((x+0, y+1)),
            sample_pixel((x+0, y-1)),
            sample_pixel((x+1, y+1)),
            sample_pixel((x-1, y-1)),
            sample_pixel((x+1, y-1)),
            sample_pixel((x-1, y+1)),
        ))
            
    segments = []
    inside = False
    run_length = 0
    gap_length = 0
    start_point = None
    end_point = None
    for point in points_in_line():
        # only requires a single pixel in the 3x3 region to be lit up.
        filled = sample_region(point) < 192
        # print '    pixel:', point, 'ON' if filled else 'OFF', run_length, gap_length
        if filled and inside:
            run_length += 1
            gap_length = 0
            end_point = point
        elif filled and not inside:
            start_point = point
            end_point = point
            inside = True
            gap_length = 0
            run_length = 0
        elif not filled and inside:
            gap_length += 1
            if gap_length > SEGMENT_FILL_GAP:
                if run_length >= SEGMENT_MIN_LENGTH:
                    if start_point != None and end_point != None:
                        segments.append({
                            "start": start_point,
                            "end": end_point,
                            "rho": rho,
                            "theta": theta,
                            "pixels": run_length,
                        })
                    else:
                        print 'start_point and end_point should be populated by this point...'
                inside = False
                run_length = 0
                gap_length = 0
                end_point = None
                start_point = None
        elif not filled and not inside:
            start_point = None
            end_point = None
            run_length = 0
            gap_length = 0

    if inside and start_point and end_point and run_length >= SEGMENT_MIN_LENGTH:
        segments.append({
            "start": start_point,
            "end": end_point,
            "rho": rho,
            "theta": theta,
            "pixels": run_length,
        })
    return segments
                


def detect(filename):
    """ applies the hough transform to detect lines,
        reduces those to segments, then saves a new
        image with the segments draw on top of the original."""

    input_filename = os.path.join('problems', filename)
    output_filename = os.path.join('solutions', filename)
    reconstruction_filename = os.path.join('reconstruction', filename)

    img = Image.open(input_filename).convert("L")
    transform = hough_transform(img)
    peaks = find_peaks(transform)

    reconstruction = img.convert('RGB')
    draw_reconstruction = ImageDraw.Draw(reconstruction)
    for peak in peaks:
        # print 'peak:', repr(peak)
        segments = find_segments(img, peak)
        for segment in segments:
            start = segment['start']
            end = segment['end']
            print '  segment:', start, '->', end, segment['pixels'], 'pixel line at', segment['theta'], 'degrees'
            draw_reconstruction.line((start, end), fill=(0,200,0))
    reconstruction.save(reconstruction_filename)

    output = transform.convert('RGB')
    draw_output = ImageDraw.Draw(output)

    # peak_descriptions = [describe(peak) for peak in peaks[0:4]]
    # description = ", ".join(sorted(peak_descriptions))
    # print filename, '->', description

    for index, (value, peak) in enumerate(peaks):
        # print filename, repr(peak), value
        for radius in range(3, 4):
            color = (255, 0, 0)  # red
            draw_output.ellipse(
                (peak[0]-radius, peak[1]-radius,
                 peak[0]+radius, peak[1]+radius),
                outline=color)
    output.save(output_filename)


if __name__ == '__main__':
    problem_filter = sys.argv[1] if len(sys.argv) > 1 else ''

    def solve_if(name):
        if name.startswith(problem_filter):
            detect(name)

    for path in glob('problems/*.*'):
        name = os.path.basename(path)
        solve_if(name)

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
import logging

# initialize loggin
logger = logging.getLogger(__name__)

# constants for the Hough transform proper
RHO_STRETCH = 2  # stretch rho dimension by a factor of two
THETA_SIZE = 180  # use 1 degree granularity

PEAK_RADIUS = 5  # compare peaks against a neighborhood of this radius
PEAK_THRESHOLD = 90  # need 10 pixels or contributing to the line

SEGMENT_FILL_GAP = 3  # can have a gap of three pixels before it splits
SEGMENT_MIN_LENGTH = 7  # segments below this length will be discarded


def mobius_wrap(size, point):
    '''given a point possibly outside a region, map it to a point within in the
    image as if the top-edge joined to bottom edge with a 180 degree twist.'''

    # TODO: wrap X too? (not super important, actually)
    if point[1] >= size[1]:
        return (size[0] + 1 - point[0], point[1] % size[1])
    elif point[1] < 0:
        return (size[0] + 1 - point[0], size[1] + point[1])
    else:
        return point


def rho_offset(img):
    '''map a real rho value, which may be negative and non-integer, to a
    strictly positive integer value that can be used as a pixel coordinate on
    the image.'''

    width, height = img.size
    return int(max(height, width) * RHO_STRETCH * sqrt(2) + 1)


def out_of_bounds(img, point):
    '''returns false iff the point is a legal pixel in the image.'''
    if point[0] < 0 or point[1] < 0:
        return True
    if point[0] >= img.size[0]:
        return True
    if point[1] >= img.size[1]:
        return True
    return False


def brighten(img, center):
    '''brightens a 3x3 pixel area in an image around a center point
    in an appoximately guassian pattern.'''

    # TODO: this effect could be acheived more efficiently by
    # applying guassian blur to the transform image afterwards.

    x, y = center

    def bump_pixel(point, vote):
        point = mobius_wrap(img.size, point)
        if not out_of_bounds(img, point):
            old_value = img.getpixel(point)
            img.putpixel(point, old_value + vote)

    # center
    bump_pixel((x, y), 9)

    # compass points
    bump_pixel((x+1, y+0), 5)
    bump_pixel((x-1, y+0), 5)
    bump_pixel((x+0, y+1), 5)
    bump_pixel((x+0, y-1), 5)

    # diagonals
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
    '''returns a list of (height, (x,y)) representing distinct local maxima
    in the image that are significantly higher than any surrounding pixels.'''

    # Each point is compared to a circular region of pixels surrounding it.
    # This region is called the neighborhood, and its members are called
    # neighbors.  the center point is NOT part of the neighborhood.  The peak
    # must be larger than all its neighbors (which is slightly stronger then
    # simply being a local maxima) and must also be significantly higher than
    # the average of its neighborhood to qualify as a "peak".

    peaks = []
    radius_squared = PEAK_RADIUS ** 2
    xy_range = range(-int(floor(PEAK_RADIUS)), int(floor(PEAK_RADIUS))+1)

    for cx in xrange(img.size[0]):
        for cy in xrange(img.size[1]):
            center = (cx, cy)
            center_height = img.getpixel(center)

            # compute the average and maximum of the neighborhood
            highest_neighbor = 0
            total_height = 0
            pixel_count = 0
            for dx in xy_range:
                for dy in xy_range:
                    neighbor = mobius_wrap(img.size, (cx+dx, cy+dy))

                    if not (dx == 0 and dy == 0):
                        if not out_of_bounds(img, neighbor):
                            if (dx**2 + dy**2) < radius_squared:
                                height = img.getpixel(neighbor)

                                # increment these for the average
                                total_height += height
                                pixel_count += 1

                                # also pick off the maximum
                                if height > highest_neighbor:
                                    highest_neighbor = height

            # determine eligibility for peak-hood
            if center_height > highest_neighbor:
                average_height = floor(total_height / pixel_count)
                relative_height = center_height - average_height
                if relative_height > PEAK_THRESHOLD:
                    peak = (relative_height, center)
                    logger.debug('peak detected: %s', repr(peak))
                    peaks.append(peak)

    return peaks


def find_segments(img, peak):
    '''returns endpoints for the segments in the line that are actually
    active).  Several hueristic are used to detect segments. 1st, because the
    matched line may not perfectly align with the original image, we check the
    3x3 grid of pixels surrounded on the point. 2nd, we allow gaps up to a
    certain threshold; if another pixel is encountered before the gap runs out,
    the segment continues. This allows for dotted/dashed lines or simply
    missing an occasional pixel.  Third, only segments above a certain length
    will be returned. This avoids returning segments for spurious points that
    happen to be in alignment with the detected feature.  '''

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

        logger.debug(
            '    pixel: %s %s %d %d',
            repr(point),
            'ON' if filled else 'OFF',
            run_length,
            gap_length)

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
                    segments.append({
                        "start": start_point,
                        "end": end_point,
                        "rho": rho,
                        "theta": theta,
                        "pixels": run_length,
                    })
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

    if inside and run_length >= SEGMENT_MIN_LENGTH:
        segments.append({
            "start": start_point,
            "end": end_point,
            "rho": rho,
            "theta": theta,
            "pixels": run_length,
        })
    return segments


def detect_line_segments(filename):
    """ applies the hough transform to detect lines,
        reduces those to segments, then saves a new
        image with the segments draw on top of the original."""

    # read and interpret the image
    input_filename = os.path.join('problems', filename)
    logger.info('detecting line segments in %s', input_filename)
    img = Image.open(input_filename).convert("L")
    transform = hough_transform(img)
    peaks = find_peaks(transform)

    # save the reconstruction as an image
    reconstruction = img.convert('RGB')
    draw_reconstruction = ImageDraw.Draw(reconstruction)
    for peak in peaks:
        logger.debug('peak: %s', repr(peak))
        segments = find_segments(img, peak)
        for segment in segments:
            start = segment['start']
            end = segment['end']

            logger.info(
                '  segment: %s -> %s, %d pixels at %d degrees',
                repr(start),
                repr(end),
                segment['pixels'],
                segment['theta'])

            # show the segment as a green line
            draw_reconstruction.line((start, end), fill=(0, 200, 0))

    reconstruction_filename = os.path.join('reconstruction', filename)
    reconstruction.save(reconstruction_filename)
    logger.info('saved %s', reconstruction_filename)

    # save the transform to an image for visual inspection

    transform_with_peaks = transform.convert('RGB')
    draw_transform_with_peaks = ImageDraw.Draw(transform_with_peaks)
    for (height, point) in peaks:
        for radius in range(3, 4):
            color = (255, 0, 0)  # red
            draw_transform_with_peaks.ellipse(
                (point[0]-radius, point[1]-radius,
                 point[0]+radius, point[1]+radius),
                outline=color)

    output = combine(quadruple(img), transform_with_peaks, quadruple(reconstruction))
    output_filename = os.path.join('solutions', filename)
    output.save(output_filename)
    logger.info('saved %s', output_filename)


def quadruple(img):
    width, height = img.size
    return img.resize( (width*4, height*4) )


def combine(*imgs):
    '''helper function to combine multiple images into one'''

    width = sum(img.size[0] + 10 for img in imgs)
    height = max(img.size[1] + 10 for img in imgs) 
    background_color = (128, 128, 128)  # grey
    combined = Image.new('RGB', (width, height), background_color)

    offset = 5
    for img in imgs:
        combined.paste(img, (offset, 5 + (height - img.size[1])//2 ))
        offset += img.size[0] + 10

    return combined 


if __name__ == '__main__':
    # display output
    logger.setLevel(logging.DEBUG)
    log_writer = logging.StreamHandler(sys.stdout)
    log_writer.setLevel(logging.INFO)
    logger.addHandler(log_writer)

    problem_filter = sys.argv[1] if len(sys.argv) > 1 else ''

    for path in glob('problems/*.*'):
        name = os.path.basename(path)
        if name.startswith(problem_filter):
            try:
                detect_line_segments(name)
            except Exception, ex:
                logger.error("Error while detecting line segments", exc_info=True)


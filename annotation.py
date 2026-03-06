from textgrid import TextGrid

def load_textgrid(filename):
    return TextGrid.fromFile(filename)

def extract_time_points(textgrid = None, filename = None):
    if textgrid is None and filename is None:
        raise ValueError('Either textgrid or filename should be provided')
    if textgrid is None: textgrid = load_textgrid(filename)
    time_points = [x.time for x in textgrid[0]]
    return time_points

def time_points_to_segments(time_points, audio_duration):
    segments = []
    time_points = [0.0] + time_points + [audio_duration]
    for i in range(len(time_points)-1):
        segments.append((time_points[i], time_points[i+1]))
    return segments
    


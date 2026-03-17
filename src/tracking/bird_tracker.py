import math


class Track:
    def __init__(self, track_id, box):
        self.id = track_id
        self.box = box
        self.missed = 0


class BirdTracker:

    def __init__(self, max_distance=80, max_missed=10):

        self.next_id = 1
        self.tracks = []

        self.max_distance = max_distance
        self.max_missed = max_missed

    def center(self, box):

        x1, y1, x2, y2 = box
        return ((x1+x2)/2, (y1+y2)/2)

    def distance(self, box1, box2):

        c1 = self.center(box1)
        c2 = self.center(box2)

        return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)

    def update(self, detections):

        updated_tracks = []

        for det in detections:

            x1, y1, x2, y2, conf = det
            box = (x1, y1, x2, y2)

            best_track = None
            best_dist = self.max_distance

            for track in self.tracks:

                d = self.distance(track.box, box)

                if d < best_dist:
                    best_dist = d
                    best_track = track

            if best_track:

                best_track.box = box
                best_track.missed = 0
                updated_tracks.append(best_track)

            else:

                new_track = Track(self.next_id, box)
                self.next_id += 1

                updated_tracks.append(new_track)

        for track in self.tracks:

            if track not in updated_tracks:

                track.missed += 1

                if track.missed < self.max_missed:
                    updated_tracks.append(track)

        self.tracks = updated_tracks

        result = []

        for t in self.tracks:

            x1, y1, x2, y2 = t.box
            result.append((x1, y1, x2, y2, t.id))

        return result
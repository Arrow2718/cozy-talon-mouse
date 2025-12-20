import sys
# try:
#     import cv2
# except ImportError:
#     print("Python module 'opencv-python-headless' not found. Can not search for boxes. Check flex-grid readme for installation instructions")
import numpy as np
import json
import base64
from numpy.typing import NDArray


# TODO: rewrite this without talon.skia.Image dependency
# def view_image(image_array, name):
#     # open the image (macOS only)
#     Image.from_array(image_array).write_file(f"/tmp/{name}.jpg")
#     subprocess.run(("open", f"/tmp/{name}.jpg"))


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.center = Point(x + w / 2, y + h / 2)


class RectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Rect):
            out_dict = obj.__dict__
            del out_dict["center"]
            return out_dict
        return json.JSONEncoder.default(self, obj)


def find_boxes_at_best_threshold(box_size_lower, box_size_upper, img):
    results = {}

    # def find_maximum(function, lower, upper, iterations_left):
    #     middle = int((upper + lower) / 2)
    #     result = len(function(middle))
    #     results[middle] = result

    #     # short circuit when out of iterations or results are all the same
    #     if iterations_left == 0 or (results[lower] == result == results[upper]):
    #         return middle

    #     # handle triangle case, e.g. 4, 10, 6
    #     if results[lower] < result > results[upper]:
    #         if results[lower] > results[upper]:
    #             return find_maximum(function, lower, middle, iterations_left - 1)
    #         else:
    #             return find_maximum(function, middle, upper, iterations_left - 1)

    #     if result > results[lower]:
    #         return find_maximum(function, middle, upper, iterations_left - 1)
    #     else:
    #         return find_maximum(function, lower, middle, iterations_left - 1)

    def find_boxes_at_threshold(threshold):
        return find_boxes(threshold, box_size_lower, box_size_upper, img)

    # first do a broad scan, checking number of boxes found across a range of thresholds
    results = {
        threshold: len(find_boxes_at_threshold(threshold))
        for threshold in range(5, 256, 15)
    }
    # print(results, file= sys.stderr)
    
    lower = 5
    upper = 5
    upper_result = results[5]
    # iterate up threshold values. when a new max is found, store threshold as upper. old upper
    # becomes lower.
    for threshold, result in results.items():
        if result >= upper_result:
            upper_result = result
            lower = upper
            upper = threshold
            # print(upper, file= sys.stderr)
            # print(lower, file= sys.stderr)
            
    final_threshold = upper        
            
            

    # final_threshold = find_maximum(find_boxes_at_threshold, lower, upper, 4)

    return final_threshold, find_boxes_at_threshold(final_threshold)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def find_boxes(
    threshold: int,
    box_size_lower: int,
    box_size_upper: int,
    img: NDArray[np.uint8],
) -> NDArray[np.int_]:
    if img.ndim != 2:
        raise ValueError("img must be a 2D grayscale array")
    if box_size_lower <= 0 or box_size_upper <= 0:
        raise ValueError("box_size_lower/upper must be positive")
    if box_size_lower >= box_size_upper:
        raise ValueError("box_size_lower must be < box_size_upper")

    def close_morph_rect_2x1(binary: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """binary close matching cv2.morphologyEx with MORPH_CLOSE over rect(2, 1)"""
        dil = binary.copy()
        dil[:, 1:] |= binary[:, :-1]
        ero = dil.copy()
        ero[:, 1:] &= dil[:, :-1]
        return ero

    def boxes_from_mask_rle(mask: NDArray[np.bool_]) -> NDArray[np.int_]:
        """
        turn a binary mask into bounding boxes for connected blobs

        1. find contiguous runs of true pixels in each row
        2. link runs to prior-row runs using 8-connectivity overlap
        3. union linked runs into components with a disjoint-set
        4. update min/max bounds on unions and current-run merges

        input mask | output boxes
        .######..  | .++++++..
        .######..  | .+....+..
        .######..  | .+....+..
        .######..  | .++++++..
        .........  | .........
        ...###...  | ...+++...
        ...###...  | ...+.+...
        ...###...  | ...+++...
        """
        h, w = mask.shape
        pad = 1

        parents: list[int] = []
        sizes: list[int] = []
        minx: list[int] = []
        miny: list[int] = []
        maxx: list[int] = []
        maxy: list[int] = []

        def uf_new(x1: int, x2: int, y: int) -> int:
            """create a new component and seed its bounds"""
            cid = len(parents)
            parents.append(cid)
            sizes.append(1)
            minx.append(x1)
            miny.append(y)
            maxx.append(x2)
            maxy.append(y)
            return cid

        def uf_find(i: int) -> int:
            """find root with path compression"""
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        def uf_union(a: int, b: int) -> int:
            """union two components and merge bounds"""
            ra = uf_find(a)
            rb = uf_find(b)
            if ra == rb:
                return ra
            if sizes[ra] < sizes[rb]:
                ra, rb = rb, ra
            parents[rb] = ra
            sizes[ra] += sizes[rb]
            minx[ra] = min(minx[ra], minx[rb])
            miny[ra] = min(miny[ra], miny[rb])
            maxx[ra] = max(maxx[ra], maxx[rb])
            maxy[ra] = max(maxy[ra], maxy[rb])
            return ra

        padded = np.zeros((h, w + 2), dtype=np.int8)
        padded[:, 1:-1] = mask.astype(np.int8)
        diff = np.diff(padded, axis=1)

        prev_starts = np.zeros((0,), dtype=int)
        prev_ends = np.zeros((0,), dtype=int)
        prev_ids = np.zeros((0,), dtype=int)
        # walk rows, build runs, and link to previous-row components
        for y in range(h):
            starts = np.flatnonzero(diff[y] == 1)
            ends = np.flatnonzero(diff[y] == -1) - 1
            n_runs = starts.size

            # overlap if current run intersects a padded previous run in x
            overlaps_mat = (
                (starts[:, None] <= (prev_ends[None, :] + pad)) &
                (ends[:, None] >= (prev_starts[None, :] - pad))
            )

            comp_ids = np.empty(n_runs, dtype=int)
            # assign each run to an existing component or create a new one
            for idx in range(n_runs):
                x1 = int(starts[idx])
                x2 = int(ends[idx])
                overlaps = prev_ids[overlaps_mat[idx]]
                if not overlaps.size:
                    comp_id = uf_new(x1, x2, y)
                else:
                    root = uf_find(int(overlaps[0]))
                    # merge any additional overlaps into the root component
                    for oid in overlaps[1:]:
                        root = uf_union(root, int(oid))
                    minx[root] = min(minx[root], x1)
                    maxx[root] = max(maxx[root], x2)
                    miny[root] = min(miny[root], y)
                    maxy[root] = max(maxy[root], y)
                    comp_id = root

                comp_ids[idx] = comp_id
            prev_starts = starts
            prev_ends = ends
            prev_ids = comp_ids

        boxes: list[tuple[int, int, int, int]] = []
        seen: set[int] = set()
        # collect unique component bounds from union-find roots
        for i in range(len(parents)):
            r = uf_find(i)
            if r in seen:
                continue
            seen.add(r)
            boxes.append((minx[r], miny[r], maxx[r], maxy[r]))
        return np.asarray(boxes, dtype=int).reshape((-1, 4))

    # threshold -> close morphology -> component boxes -> size/shape filter -> dedup
    binary: NDArray[np.bool_] = img > np.uint8(threshold)
    morph: NDArray[np.bool_] = close_morph_rect_2x1(binary)
    fg_boxes = boxes_from_mask_rle(morph)
    if fg_boxes.size == 0:
        return fg_boxes

    ws = fg_boxes[:, 2] - fg_boxes[:, 0] + 1
    hs = fg_boxes[:, 3] - fg_boxes[:, 1] + 1
    size_ok = (
        (ws >= box_size_lower)
        & (ws < box_size_upper)
        & (hs > box_size_lower)
        & (hs < box_size_upper)
        & (np.abs(ws - hs) < 0.8 * ws)
    )
    if not np.any(size_ok):
        return fg_boxes[:0]

    arr = np.stack((fg_boxes[:, 0], fg_boxes[:, 1], ws, hs), axis=1)[size_ok]
    cx = arr[:, 0] + (arr[:, 2] / 2.0)
    cy = arr[:, 1] + (arr[:, 3] / 2.0)
    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])
    near = (dx < box_size_lower) & (dy < box_size_lower)
    omit = np.triu(near, k=1).any(axis=1)
    
    return arr[~omit].tolist()


if __name__ == "__main__":
    args = json.load(sys.stdin)

    # convert base64 string to numpy array
    img_b64 = base64.b64decode(args["img"])
    img = np.frombuffer(img_b64, dtype=np.uint8)

    # reshape array to image dimensions
    img = img.reshape(args["height"], args["width"], 3)

    # convert to grayscale
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = rgb2gray(img)

    threshold = args["threshold"]
    if threshold >= 0:
        boxes = find_boxes(
            threshold, args["box_size_lower"], args["box_size_upper"], img
        )
    else:
        threshold, boxes = find_boxes_at_best_threshold(
            args["box_size_lower"], args["box_size_upper"], img
        )

        
    # print output as json
    output = {"boxes": [], "threshold": threshold}
    
    for box in boxes:
    
        rect = Rect( box[0], box[1], box[2], box[3])
        output["boxes"].append(rect)
    
    print(json.dumps(output, cls=RectEncoder, separators=(",", ":")))

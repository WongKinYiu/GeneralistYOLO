DET_LAYERS = [
    'Detect', 'DualDetect', 'TripleDetect', 'DDetect', 'DualDDetect', 'TripleDDetect',
    'Segment', 'DSegment', 'DualDSegment',
    'Panoptic', 'DPanoptic', 'DualDPanoptic',
]

CAP_LAYERS = [
    'Grit', 'DualGrit',
]

OUTPUT_LAYERS = [
    'OutputLayer',
]

def find_layer(model, layers = []):
    layer_idx = None
    if 0 != len(layers):
        for i in range(len(model) - 1, -1, -1):  # backword search
            if model[i].__class__.__name__ in layers:
                layer_idx = i
                break

    return layer_idx

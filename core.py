"""Kitchen review rules. Coordinates use original-image pixels."""
import math

def box(value, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('Each box must contain four coordinates.')
    x1, y1, x2, y2 = map(float, value)
    if not all(math.isfinite(v) for v in (x1,y1,x2,y2)):
        raise ValueError('Coordinates must be finite.')
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Box must be inside the image with positive width and height.')
    return [x1,y1,x2,y2]

def review(detections, zone, width, height, threshold=0.25, prep=None, scoped=False):
    zone = box(zone, width, height) if zone else None
    prep = box(prep, width, height) if prep else None
    events = []
    for i, item in enumerate(detections):
        b = box(item['box'], width, height)
        label = str(item['label']).strip().lower()
        if not label:
            raise ValueError('Object labels cannot be empty.')
        if label == 'person':
            cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
            if scoped and (not prep or not (prep[0]<=cx<=prep[2] and prep[1]<=cy<=prep[3])):
                continue
            status = item.get('hairnet', 'uncertain')
            if status not in ('present','absent','uncertain'):
                raise ValueError('Invalid hairnet review status.')
            origin=item.get('assessment_source','manual')
            detail=('Model prediction; verify visually' if origin=='trained_model' else 'Manual visual assessment') if status!='uncertain' else 'Hairnet not assessed; human review required'
            if origin=='illustration':detail='Illustrative scenario label; not a model prediction'
            events.append(dict(object=i+1, type='hairnet_review', label=label,
                status=status, detail=detail, box=b))
        elif zone and label not in ('hairnet','no_hairnet','uncertain_head','sink','oven','refrigerator','microwave','dining table'):
            if item.get('floor_contact')=='no':continue
            # Bottom 20% approximates the floor footprint, not the entire object silhouette.
            foot = [b[0], b[3]-(b[3]-b[1])*.2, b[2], b[3]]
            area = (foot[2]-foot[0])*(foot[3]-foot[1])
            overlap = max(0,min(foot[2],zone[2])-max(foot[0],zone[0])) * max(0,min(foot[3],zone[3])-max(foot[1],zone[1]))
            fraction = overlap/area
            if fraction >= threshold:
                events.append(dict(object=i+1,type='possible_obstruction',label=label,status='review',
                    detail=f'{fraction:.0%} of estimated footprint overlaps walkway; verify floor contact',box=b))
    return events

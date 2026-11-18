"""Conservative assignment of explicit head-class detections to person boxes."""
def associate(people, heads, min_conf=.65):
    assignments={i:[] for i,d in enumerate(people) if d['label']=='person'}
    for head in heads:
        if head['label'] not in ('hairnet','no_hairnet') or head.get('confidence',0)<min_conf:
            continue
        x1,y1,x2,y2=head['box']; cx=(x1+x2)/2; cy=(y1+y2)/2
        candidates=[]
        for i in assignments:
            px,py,qx,qy=people[i]['box']
            if px<=cx<=qx and py<=cy<=py+(qy-py)*.45:
                candidates.append(i)
        if len(candidates)==1: assignments[candidates[0]].append(head)
    for i,heads in assignments.items():
        labels={h['label'] for h in heads}
        people[i]['hairnet']='uncertain'
        people[i]['assessment_source']='not_assessed'
        if len(labels)==1:
            people[i]['hairnet']='present' if 'hairnet' in labels else 'absent'
            people[i]['assessment_source']='trained_model'
    return people

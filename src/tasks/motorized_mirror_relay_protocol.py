"""Hardware-independent action contract for lab-matched relay experiments."""

import json


STATION_IDS = tuple(f'C{index}' for index in range(1, 9))


def parse_action(text, station, offsets=None):
    action = json.loads(text)
    if not isinstance(action, dict) or not isinstance(action.get('reason'), str) or not action['reason'].strip():
        raise ValueError('Action needs a reason')
    kind = action.get('action')
    keys = ({'action', 'reason', 'mirror', 'vertical', 'horizontal'} if kind == 'tilt'
            else {'action', 'reason', 'station'} if kind == 'measure' else {'action', 'reason'})
    if set(action) != keys:
        raise ValueError('Unexpected or missing action fields')
    if kind == 'tilt':
        if action['mirror'] not in ('M1', 'M2', 'M3', 'M4'):
            raise ValueError('Choose mirror M1-M4')
        mirror = int(action['mirror'][1])
        for label, axis in (('vertical', 'in_plane'), ('horizontal', 'out_of_plane')):
            delta = action[label]
            if type(delta) is not int:
                raise ValueError(f'M{mirror} {label}: delta must be an integer')
            current = offsets[f'mirror_{mirror}_{axis}_steps'] if offsets is not None else 0
            low, high = max(-15000, -30000-current), min(15000, 30000-current)
            if not low <= delta <= high:
                raise ValueError(f'M{mirror} {label}: current commanded total {current:+d}, '
                                 f'requested delta {delta:+d}, resulting total {current+delta:+d}. '
                                 f'Allowed delta range [{low}, {high}]; per-move limit +/-15000, '
                                 'cumulative limit +/-30000 steps.')
    elif kind == 'measure' and action['station'] in STATION_IDS:
        pass
    elif kind == 'done' and station == 'C8':
        pass
    else:
        raise ValueError('Invalid action for this station')
    return action


def valid_action(text, station, offsets=None):
    try:
        parse_action(text, station, offsets)
        return True
    except (ValueError, TypeError, KeyError):
        return False


def feedback_validator(llm, station, offsets, record, save):
    def verify(text):
        try:
            parse_action(text, station, offsets)
            return True
        except (ValueError, TypeError, KeyError) as exc:
            feedback = {'rejected_action': text, 'error': str(exc),
                        'instruction': 'No movement was executed. Using the same image and state, submit one corrected action.'}
            record.setdefault('rejections', []).append(feedback)
            save()
            llm.add_user_message(json.dumps(feedback))
            return False
    return verify

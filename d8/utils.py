
def compact_repr(a, max_n=10):
    assert isinstance(a, (tuple, list))
    body = [str(i) for i in a]
    if len(a) > max_n:
        body = body[:max_n//2] + ['...'] + body[-max_n//2:]
    body = ', '.join(body)
    if isinstance(a, tuple):
        return f'size={len(a)}, ({body})'
    return f'size={len(a)}, [{body}]'

import site, os

patched_any = False
for p in site.getsitepackages():
    f = os.path.join(p, 'torch', 'utils', 'tensorboard', '_embedding.py')
    if os.path.exists(f):
        txt = open(f, encoding='utf-8').read()
        if '_HAS_GFILE_JOIN = hasattr(tf.io.gfile, "join")' in txt:
            patched = txt.replace(
                '_HAS_GFILE_JOIN = hasattr(tf.io.gfile, "join")',
                'try:\n    _HAS_GFILE_JOIN = hasattr(tf.io.gfile, "join")\nexcept Exception:\n    _HAS_GFILE_JOIN = False'
            )
            open(f, 'w', encoding='utf-8').write(patched)
            print('Patched:', f)
            patched_any = True
        else:
            print('Already patched or line not found in:', f)

if not patched_any:
    print('No matching file found in site-packages.')
    print('Searched:', site.getsitepackages())
